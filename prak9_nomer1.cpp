// ============================================================================
// Praktikum 9 - Template Matching (PERCOBAAN 1) - VERSI SLIDER / TRACKBAR
//
// Ganti metode cukup GESER trackbar (seperti modul asli), tidak perlu enter.
//   - Template dipilih OTOMATIS (crop dari sumber, simpan PNG -> piksel identik).
//   - Label metode tampil jelas di layar (0: SQDIFF, dst, gaya laporan teman)
//     dan di judul window.
//   - Tekan 's' untuk menyimpan tampilan metode yang sedang aktif (out_<metode>.png).
//   - Tekan ESC untuk keluar.
//
// Build (VS/Windows): Build & Run.
// Build (Linux): g++ percobaan1_slider.cpp -o tm `pkg-config --cflags --libs opencv4`
// ============================================================================

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
using namespace cv;
using namespace std;

string image_window = "Source Image";
string result_window = "Result window";
string outDir = "C:/Users/riyadh/Downloads/";

Mat img, templ;
int match_method = 0;
int max_Trackbar = 5;

struct MethodInfo { int id; string name; string label; bool useMin; };
vector<MethodInfo> METHODS = {
    { TM_SQDIFF,        "0_SQDIFF",          "0: SQDIFF",           true  },
    { TM_SQDIFF_NORMED, "1_SQDIFF_NORMED",   "1: SQDIFF NORMED",    true  },
    { TM_CCORR,         "2_TM_CCORR",        "2: TM CCORR",         false },
    { TM_CCORR_NORMED,  "3_TM_CCORR_NORMED", "3: TM CCORR NORMED",  false },
    { TM_CCOEFF,        "4_TM_COEFF",        "4: TM COEFF",         false },
    { TM_CCOEFF_NORMED, "5_TM_COEFF_NORMED", "5: TM COEFF NORMED",  false },
};

Mat g_lastCombo;   // tampilan terakhir (untuk disimpan saat tekan 's')

// Pilih kotak template otomatis: jendela dengan energi gradien tertinggi.
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

// Callback trackbar: dipanggil tiap kali slider digeser.
void MatchingMethod(int, void*)
{
    const MethodInfo& m = METHODS[match_method];

    Mat result; matchTemplate(img, templ, result, m.id);
    double mn, mx; Point mnL, mxL, matchLoc;
    minMaxLoc(result, &mn, &mx, &mnL, &mxL, Mat());
    double rawScore;
    if (m.useMin) { matchLoc = mnL; rawScore = mn; }
    else { matchLoc = mxL; rawScore = mx; }

    // Source + kotak hitam
    Mat srcDisp; img.copyTo(srcDisp);
    rectangle(srcDisp, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows),
        Scalar::all(0), 2, 8, 0);

    // Result map ternormalisasi + kotak
    Mat resNorm; normalize(result, resNorm, 0, 1, NORM_MINMAX, -1, Mat());
    Mat resDisp = resNorm.clone();
    rectangle(resDisp, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows),
        Scalar::all(0), 2, 8, 0);

    // Label metode jelas di layar (band hitam + teks kuning)
    Mat srcShow = srcDisp.clone();
    rectangle(srcShow, Point(0, 0), Point(srcShow.cols, 42), Scalar(0, 0, 0), FILLED);
    putText(srcShow, m.label, Point(12, 30), FONT_HERSHEY_SIMPLEX, 0.85,
        Scalar(0, 255, 255), 2, LINE_AA);

    setWindowTitle(image_window, "Source Image  -  " + m.label);
    setWindowTitle(result_window, "Result window -  " + m.label);

    imshow(image_window, srcShow);
    imshow(result_window, resDisp);

    // Siapkan gabungan berlabel untuk disimpan bila perlu (tekan 's')
    Mat res8, resBGR; resNorm.convertTo(res8, CV_8U, 255.0);
    cvtColor(res8, resBGR, COLOR_GRAY2BGR);
    rectangle(resBGR, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows),
        Scalar::all(0), 2, 8, 0);
    resize(resBGR, resBGR, Size(img.cols, img.rows));
    hconcat(srcDisp, resBGR, g_lastCombo);
    rectangle(g_lastCombo, Point(0, 0), Point(g_lastCombo.cols, 30), Scalar(0, 0, 0), FILLED);
    putText(g_lastCombo, m.label + "   score(raw)=" + format("%.4g", rawScore),
        Point(10, 21), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1, LINE_AA);

    printf("%-20s | match=(%d,%d) | score=%.6g\n", m.label.c_str(),
        matchLoc.x, matchLoc.y, rawScore);
}

int main()
{
    string srcPath = outDir + "pejuangkemerdekaan.jpg";
    img = imread(srcPath, IMREAD_COLOR);
    if (img.empty()) { cerr << "ERROR: sumber tidak ditemukan: " << srcPath << endl; return -1; }

    // Template otomatis
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
        if (key == 27) break;                       // ESC
        if (key == 's' || key == 'S') {             // simpan tampilan aktif
            string f = outDir + "out_" + METHODS[match_method].name + ".png";
            imwrite(f, g_lastCombo);
            cout << "Tersimpan: " << f << "\n";
        }
    }
    destroyAllWindows();
    return 0;
}
