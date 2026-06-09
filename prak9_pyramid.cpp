// ============================================================================
// Praktikum 9 - PERCOBAAN 2 - IMAGE PYRAMID (gaya laporan teman / Program 3)
//
// Output seperti laporan teman:
//   - Satu KOTAK HIJAU pada match terbaik.
//   - Teks "Score: ..." (merah) di pojok kiri atas.
//   - Window "Source Image" + "Result Window".
//   - Ganti metode lewat SLIDER (trackbar).
//
// Multi-skala: template dibuat dalam beberapa skala via pyrUp/pyrDown,
// dicocokkan di tiap skala, lalu skala dengan skor terbaik yang dipakai.
//
// Template dipilih OTOMATIS (crop dari sumber, simpan PNG) supaya akurat.
//   - Geser slider untuk ganti metode (0..5).
//   - Tekan 's' untuk simpan tampilan aktif (pyr_<metode>.png).
//   - Tekan ESC untuk keluar.
//
// Build (VS/Windows): Build & Run.
// Build (Linux): g++ percobaan2_pyramid_seperti_teman.cpp -o tm2 `pkg-config --cflags --libs opencv4`
// ============================================================================

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
using namespace cv;
using namespace std;

string image_window = "Source Image";
string result_window = "Result Window";
string outDir = "C:/Users/riyadh/Downloads/";

Mat img, templ, result;
int match_method = 0;
int max_Trackbar = 5;
int maxpyrlevel = 5;   // jumlah level pyramid (seperti Program 3 teman)

struct MethodInfo { int id; string name; string label; bool useMin; bool normed; };
vector<MethodInfo> METHODS = {
    { TM_SQDIFF,        "0_SQDIFF",          "0: SQDIFF",           true,  false },
    { TM_SQDIFF_NORMED, "1_SQDIFF_NORMED",   "1: SQDIFF NORMED",    true,  true  },
    { TM_CCORR,         "2_TM_CCORR",        "2: TM CCORR",         false, false },
    { TM_CCORR_NORMED,  "3_TM_CCORR_NORMED", "3: TM CCORR NORMED",  false, true  },
    { TM_CCOEFF,        "4_TM_COEFF",        "4: TM COEFF",         false, false },
    { TM_CCOEFF_NORMED, "5_TM_COEFF_NORMED", "5: TM COEFF NORMED",  false, true  },
};

Mat g_lastDisplay;

Rect autoTemplateRect(const Mat& src, int side)
{
    Mat gray, gx, gy, mag;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    Sobel(gray, gx, CV_32F, 1, 0, 3);
    Sobel(gray, gy, CV_32F, 0, 1, 3);
    magnitude(gx, gy, mag);
    Mat integ; integral(mag, integ, CV_64F);
    double best = -1.0; Rect bestR(0, 0, side, side);
    const int step = 6;
    for (int y = 0; y + side <= src.rows; y += step)
        for (int x = 0; x + side <= src.cols; x += step) {
            double s = integ.at<double>(y + side, x + side) - integ.at<double>(y, x + side)
                - integ.at<double>(y + side, x) + integ.at<double>(y, x);
            if (s > best) { best = s; bestR = Rect(x, y, side, side); }
        }
    return bestR;
}

// Callback trackbar: multi-scale matching, ambil match terbaik (gaya Program 3)
void MatchingMethod(int, void*)
{
    const MethodInfo& m = METHODS[match_method];
    Mat img_display; img.copyTo(img_display);

    bool first = true; double bestComp = 0;
    Point bestLoc; Size bestSize; Mat bestRaw;

    for (int i = 0; i < maxpyrlevel; i++)
    {
        // Skala template via pyramid
        Mat tempPyr;
        if (i == 0)            tempPyr = templ.clone();   // skala asli
        else if (i % 2 == 1)   pyrUp(templ, tempPyr);     // perbesar
        else                   pyrDown(templ, tempPyr);   // perkecil

        if (tempPyr.cols > img.cols || tempPyr.rows > img.rows) continue;
        if (tempPyr.cols < 4 || tempPyr.rows < 4) continue;

        // Skor MENTAH (jangan dinormalisasi dulu, supaya bisa dibandingkan antar-skala)
        Mat res; matchTemplate(img, tempPyr, res, m.id);
        double mn, mx; Point mnL, mxL;
        minMaxLoc(res, &mn, &mx, &mnL, &mxL, Mat());

        double rawPeak = m.useMin ? mn : mx;
        Point  loc = m.useMin ? mnL : mxL;
        // skor sebanding antar-skala: normed apa adanya, mentah dibagi luas (per-piksel)
        double comp = m.normed ? rawPeak : rawPeak / (double)(tempPyr.cols * tempPyr.rows);

        bool better = first || (m.useMin ? (comp < bestComp) : (comp > bestComp));
        if (better) {
            first = false; bestComp = comp;
            bestLoc = loc; bestSize = tempPyr.size();
            res.copyTo(bestRaw);
        }
    }

    // Kotak hijau pada match terbaik (tanpa teks skor)
    rectangle(img_display, bestLoc,
        Point(bestLoc.x + bestSize.width, bestLoc.y + bestSize.height),
        Scalar(0, 255, 0), 2);

    // Peta hasil skala pemenang untuk Result Window
    Mat bestNorm; normalize(bestRaw, bestNorm, 0, 1, NORM_MINMAX, -1, Mat());

    setWindowTitle(image_window, "Source Image  -  " + m.label);
    setWindowTitle(result_window, "Result Window -  " + m.label);
    imshow(image_window, img_display);
    imshow(result_window, bestNorm);

    g_lastDisplay = img_display.clone();
    printf("%-20s | match=(%d,%d) | size=%dx%d\n",
        m.label.c_str(), bestLoc.x, bestLoc.y, bestSize.width, bestSize.height);
}

int main()
{
    string srcPath = outDir + "pejuangkemerdekaan.jpg";
    img = imread(srcPath, IMREAD_COLOR);
    if (img.empty()) { cerr << "ERROR: sumber tidak ditemukan: " << srcPath << endl; return -1; }

    int side = max(48, min(img.cols, img.rows) / 6);
    Rect r = autoTemplateRect(img, side);
    templ = img(r).clone();
    imwrite(outDir + "template_auto.png", templ);
    cout << "[AUTO] Template di " << r << " (" << templ.cols << "x" << templ.rows << ")\n";

    namedWindow(image_window, WINDOW_AUTOSIZE);
    namedWindow(result_window, WINDOW_AUTOSIZE);

    string trackbar_label =
        "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR "
        "\n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
    createTrackbar(trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod);

    MatchingMethod(0, 0);
    cout << "Geser slider untuk ganti metode. Tekan 's' untuk simpan, ESC untuk keluar.\n";

    while (true) {
        int key = waitKey(0);
        if (key == 27) break;
        if (key == 's' || key == 'S') {
            string f = outDir + "pyr_" + METHODS[match_method].name + ".png";
            imwrite(f, g_lastDisplay);
            cout << "Tersimpan: " << f << "\n";
        }
    }
    destroyAllWindows();
    return 0;
}
