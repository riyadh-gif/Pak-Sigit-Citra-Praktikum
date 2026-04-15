#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>
#include <iostream>
#include <direct.h>

using namespace cv;
using namespace std;

// ============================================================
// UTILITAS
// ============================================================
Mat addLabel(const Mat& img, const string& label) {
    Mat out;
    copyMakeBorder(img, out, 0, 50, 0, 0, BORDER_CONSTANT, Scalar(10, 10, 10));
    putText(out, label, Point(7, img.rows + 34),
        FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 4);
    putText(out, label, Point(7, img.rows + 34),
        FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 230, 255), 2);
    return out;
}

// ============================================================
// PERCOBAAN AWAL: Histogram RGB via calcHist (fungsi OpenCV)
// ============================================================
Mat drawHistogramRGB(const Mat& src) {
    vector<Mat> bgr_planes;
    split(src, bgr_planes);

    int   histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool  uniform = true, accumulate = false;

    Mat b_hist, g_hist, r_hist;
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(20, 20, 20));

    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++) {
        line(histImage,
            Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
            Scalar(255, 80, 80), 2);
        line(histImage,
            Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
            Scalar(80, 255, 80), 2);
        line(histImage,
            Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
            Scalar(80, 80, 255), 2);
    }

    // Legenda
    putText(histImage, "B", Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 80, 80), 2);
    putText(histImage, "G", Point(30, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(80, 255, 80), 2);
    putText(histImage, "R", Point(50, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(80, 80, 255), 2);
    putText(histImage, "Histogram RGB (calcHist)", Point(10, hist_h - 10),
        FONT_HERSHEY_SIMPLEX, 0.55, Scalar(200, 200, 200), 1);

    return histImage;
}

// ============================================================
// PERCOBAAN 1: Histogram Grayscale via calcHist (OpenCV)
// ============================================================
Mat drawHistogramGray_CV(const Mat& src) {
    Mat gray;
    if (src.channels() == 3) cvtColor(src, gray, COLOR_BGR2GRAY);
    else gray = src.clone();

    int   histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool  uniform = true, accumulate = false;

    Mat hist;
    calcHist(&gray, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(20, 20, 20));

    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++) {
        line(histImage,
            Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
            Scalar(220, 220, 220), 2);
    }

    putText(histImage, "Histogram Grayscale (calcHist)", Point(10, hist_h - 10),
        FONT_HERSHEY_SIMPLEX, 0.55, Scalar(200, 200, 200), 1);

    return histImage;
}

// ============================================================
// PERCOBAAN 2: Histogram Grayscale MANUAL (tanpa calcHist/normalize)
// ============================================================
Mat drawHistogramGray_Manual(const Mat& src) {
    Mat gray;
    if (src.channels() == 3) cvtColor(src, gray, COLOR_BGR2GRAY);
    else gray = src.clone();

    // Hitung frekuensi tiap nilai intensitas [0-255] secara manual
    int freq[256] = { 0 };
    for (int y = 0; y < gray.rows; y++)
        for (int x = 0; x < gray.cols; x++)
            freq[gray.at<uchar>(y, x)]++;

    // Cari nilai maksimum untuk normalisasi manual
    int maxFreq = 0;
    for (int i = 0; i < 256; i++)
        if (freq[i] > maxFreq) maxFreq = freq[i];

    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / 256);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(20, 20, 20));

    // Normalisasi manual: scale ke [0, hist_h]
    for (int i = 1; i < 256; i++) {
        int h0 = (int)((double)freq[i - 1] / maxFreq * hist_h);
        int h1 = (int)((double)freq[i] / maxFreq * hist_h);
        line(histImage,
            Point(bin_w * (i - 1), hist_h - h0),
            Point(bin_w * (i), hist_h - h1),
            Scalar(100, 255, 180), 2);
    }

    putText(histImage, "Histogram Grayscale (MANUAL - tanpa calcHist)", Point(10, hist_h - 10),
        FONT_HERSHEY_SIMPLEX, 0.52, Scalar(100, 255, 180), 1);

    return histImage;
}

// ============================================================
// MAIN
// ============================================================
int main() {
    FILE* devNull = nullptr;
    freopen_s(&devNull, "nul", "w", stderr);

    string imgPath = "C:\\Users\\riyadh\\source\\repos\\OpenCV_Test\\OpenCV_Test\\Folder Gambar\\picture_p4.bmp";
    Mat src = imread(imgPath, IMREAD_COLOR);

    FILE* con = nullptr;
    freopen_s(&con, "CON", "w", stderr);

    if (src.empty()) {
        cerr << "ERROR: Gagal membuka gambar!\nPath: " << imgPath << "\n";
        system("pause"); return -1;
    }

    string outDir = "C:\\Users\\riyadh\\source\\repos\\OpenCV_Test\\OpenCV_Test\\Folder Gambar\\hasil_praktikum7";
    _mkdir(outDir.c_str());

    cout << "========================================\n";
    cout << "  PRAKTIKUM 7 - HISTOGRAM\n";
    cout << "========================================\n\n";

    // ── Hitung semua histogram ────────────────────────────────
    Mat histRGB = drawHistogramRGB(src);
    Mat histGrayCV = drawHistogramGray_CV(src);
    Mat histGrayMan = drawHistogramGray_Manual(src);

    // Resize gambar asli untuk tile
    Mat srcThumb;
    resize(src, srcThumb, Size(512, 400));

    // Simpan
    imwrite(outDir + "\\hist_rgb.bmp", histRGB);
    imwrite(outDir + "\\hist_gray_cv.bmp", histGrayCV);
    imwrite(outDir + "\\hist_gray_manual.bmp", histGrayMan);
    imwrite(outDir + "\\gambar_asli.bmp", srcThumb);
    cout << "Semua histogram tersimpan.\n\n";

    // ── TAMPILKAN ─────────────────────────────────────────────
    cout << "========================================\n";
    cout << "Menampilkan hasil...\n";
    cout << "[ENTER] untuk lanjut\n";
    cout << "========================================\n\n";

    // --- Tampilan 1: Gambar asli + Histogram RGB ---
    Mat view1;
    hconcat(vector<Mat>{
        addLabel(srcThumb, "Gambar Asli (BGR)"),
            addLabel(histRGB, "Histogram RGB (calcHist)")
    }, view1);

    // Header
    Mat header1(56, view1.cols, CV_8UC3, Scalar(20, 20, 20));
    putText(header1, "HISTOGRAM RGB - via calcHist OpenCV",
        Point(10, 38), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255, 200, 0), 2);
    Mat show1;
    vconcat(header1, view1, show1);
    imwrite(outDir + "\\tampilan_rgb.bmp", show1);

    namedWindow("Histogram RGB", WINDOW_NORMAL);
    resizeWindow("Histogram RGB", 1050, 520);
    imshow("Histogram RGB", show1);
    cout << ">> Histogram RGB ditampilkan. Screenshot lalu tekan ENTER.\n";
    waitKey(0);
    destroyWindow("Histogram RGB");

    // --- Tampilan 2: Grayscale CV vs Manual ---
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    Mat grayBGR, grayThumb;
    cvtColor(gray, grayBGR, COLOR_GRAY2BGR);
    resize(grayBGR, grayThumb, Size(512, 400));

    Mat view2;
    hconcat(vector<Mat>{
        addLabel(grayThumb, "Gambar Grayscale"),
            addLabel(histGrayCV, "Histogram Gray (calcHist)"),
            addLabel(histGrayMan, "Histogram Gray (MANUAL)")
    }, view2);

    Mat header2(56, view2.cols, CV_8UC3, Scalar(20, 20, 20));
    putText(header2, "PERCOBAAN 1 & 2: Histogram Grayscale - CV vs Manual",
        Point(10, 38), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255, 200, 0), 2);
    Mat show2;
    vconcat(header2, view2, show2);
    imwrite(outDir + "\\tampilan_grayscale.bmp", show2);

    namedWindow("Histogram Grayscale", WINDOW_NORMAL);
    resizeWindow("Histogram Grayscale", 1550, 520);
    imshow("Histogram Grayscale", show2);
    cout << ">> Histogram Grayscale ditampilkan. Screenshot lalu tekan ENTER.\n";
    waitKey(0);
    destroyWindow("Histogram Grayscale");

    cout << "\n========================================\n";
    cout << "Semua hasil disimpan di:\n" << outDir << "\n";
    cout << "========================================\n";
    system("pause");
    return 0;
}
