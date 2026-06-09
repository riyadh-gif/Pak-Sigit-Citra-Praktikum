// ============================================================================
// Praktikum 9 - Template Matching (PERCOBAAN 1) - AUTO + STEP-BY-STEP
//
// Template dipilih OTOMATIS (tak perlu drag mouse), tetapi hasil ditampilkan
// SATU METODE per layar. Tekan ENTER / sembarang tombol untuk lanjut ke metode
// berikutnya (ESC untuk berhenti). Tiap metode tetap disimpan otomatis ke
// folder Downloads, jadi tidak wajib screenshot manual.
//
// Output di C:\Users\riyadh\Downloads\:
//   out_0_SQDIFF.png ... out_5_TM_COEFF_NORMED.png  + template_auto.png
//
// Build (Visual Studio / Windows): Build & Run.
// Build (Linux): g++ percobaan1_auto_step.cpp -o tm `pkg-config --cflags --libs opencv4`
// ============================================================================

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
using namespace cv;
using namespace std;

string image_window = "Source Image";
string result_window = "Result window";

struct MethodInfo { int id; string name; string label; bool useMin; };
vector<MethodInfo> METHODS = {
    { TM_SQDIFF,        "0_SQDIFF",          "0: SQDIFF",            true  },
    { TM_SQDIFF_NORMED, "1_SQDIFF_NORMED",   "1: SQDIFF NORMED",     true  },
    { TM_CCORR,         "2_TM_CCORR",        "2: TM CCORR",          false },
    { TM_CCORR_NORMED,  "3_TM_CCORR_NORMED", "3: TM CCORR NORMED",   false },
    { TM_CCOEFF,        "4_TM_COEFF",        "4: TM COEFF",          false },
    { TM_CCOEFF_NORMED, "5_TM_COEFF_NORMED", "5: TM COEFF NORMED",   false },
};

// Pilih kotak template otomatis: jendela dengan energi gradien (detail) tertinggi.
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

int main()
{
    string outDir = "C:/Users/riyadh/Downloads/";
    string srcPath = outDir + "pejuangkemerdekaan.jpg";

    Mat img = imread(srcPath, IMREAD_COLOR);
    if (img.empty()) { cerr << "ERROR: sumber tidak ditemukan: " << srcPath << endl; return -1; }

    // Template otomatis (crop langsung dari sumber -> piksel identik)
    int side = max(48, min(img.cols, img.rows) / 6);
    Rect r = autoTemplateRect(img, side);
    Mat templ = img(r).clone();
    imwrite(outDir + "template_auto.png", templ);
    cout << "[AUTO] Template di " << r << " (" << templ.cols << "x" << templ.rows << ")\n\n";

    namedWindow(image_window, WINDOW_AUTOSIZE);
    namedWindow(result_window, WINDOW_AUTOSIZE);

    cout << "Tekan ENTER / sembarang tombol untuk lanjut. ESC untuk berhenti.\n";
    cout << "-------------------------------------------------------------\n";

    for (size_t k = 0; k < METHODS.size(); k++)
    {
        const MethodInfo& m = METHODS[k];

        // Matching + skor mentah
        Mat result;
        matchTemplate(img, templ, result, m.id);
        double minVal, maxVal; Point minLoc, maxLoc, matchLoc;
        minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
        double rawScore;
        if (m.useMin) { matchLoc = minLoc; rawScore = minVal; }
        else { matchLoc = maxLoc; rawScore = maxVal; }

        // Tampilan source (kotak hitam)
        Mat srcDisp; img.copyTo(srcDisp);
        rectangle(srcDisp, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows),
            Scalar::all(0), 2, 8, 0);

        // Tampilan result map (ternormalisasi) + kotak
        Mat resNorm;
        normalize(result, resNorm, 0, 1, NORM_MINMAX, -1, Mat());
        Mat resDisp = resNorm.clone();
        rectangle(resDisp, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows),
            Scalar::all(0), 2, 8, 0);

        // --- Label nama metode JELAS di layar (band hitam + teks kuning) ---
        Mat srcShow = srcDisp.clone();
        rectangle(srcShow, Point(0, 0), Point(srcShow.cols, 42), Scalar(0, 0, 0), FILLED);
        putText(srcShow, m.label, Point(12, 30),
            FONT_HERSHEY_SIMPLEX, 0.85, Scalar(0, 255, 255), 2, LINE_AA);

        // Nama metode juga di judul kedua window
        setWindowTitle(image_window, "Source Image  -  " + m.label);
        setWindowTitle(result_window, "Result window -  " + m.label);

        imshow(image_window, srcShow);
        imshow(result_window, resDisp);

        // Simpan gabungan berlabel (source | result) untuk laporan
        Mat res8, resBGR; resNorm.convertTo(res8, CV_8U, 255.0);
        cvtColor(res8, resBGR, COLOR_GRAY2BGR);
        rectangle(resBGR, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows),
            Scalar::all(0), 2, 8, 0);
        resize(resBGR, resBGR, Size(img.cols, img.rows));
        Mat combo; hconcat(srcDisp, resBGR, combo);
        rectangle(combo, Point(0, 0), Point(combo.cols, 30), Scalar(0, 0, 0), FILLED);
        putText(combo, m.label + "   score(raw)=" + format("%.4g", rawScore),
            Point(10, 21), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1, LINE_AA);
        imwrite(outDir + "out_" + m.name + ".png", combo);

        printf("[%zu/6] %-20s | match=(%d,%d) | score=%.6g  -> ENTER utk lanjut\n",
            k + 1, m.name.c_str(), matchLoc.x, matchLoc.y, rawScore);

        int key = waitKey(0);             // tunggu tombol
        if (key == 27) { cout << "Dihentikan.\n"; break; }   // ESC
    }

    cout << "\nSelesai. File tersimpan di " << outDir << " (out_*.png).\n";
    destroyAllWindows();
    return 0;
}
