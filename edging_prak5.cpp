#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>
#include <iostream>
#include <direct.h>
#include <cmath>

using namespace cv;
using namespace std;

// ============================================================
// UTILITAS
// ============================================================
Mat addLabel(const Mat& img, const string& label) {
    Mat out;
    copyMakeBorder(img, out, 0, 50, 0, 0, BORDER_CONSTANT, Scalar(10, 10, 10));
    putText(out, label, Point(7, img.rows + 34),
        FONT_HERSHEY_SIMPLEX, 0.58, Scalar(0, 0, 0), 4);
    putText(out, label, Point(7, img.rows + 34),
        FONT_HERSHEY_SIMPLEX, 0.58, Scalar(0, 230, 255), 2);
    return out;
}

Mat makeMosaic(const vector<pair<string, Mat>>& tiles, const string& judul) {
    if (tiles.empty()) return Mat();
    int W = tiles[0].second.cols;
    int H = tiles[0].second.rows;
    int labelH = 50, headerH = 56;
    int cols = (int)tiles.size();

    Mat mosaic(headerH + H + labelH, W * cols, CV_8UC3, Scalar(30, 30, 30));
    rectangle(mosaic, Point(0, 0), Point(mosaic.cols, headerH), Scalar(20, 20, 20), FILLED);
    putText(mosaic, judul, Point(10, 38),
        FONT_HERSHEY_SIMPLEX, 0.78, Scalar(255, 200, 0), 2);

    for (int i = 0; i < cols; i++) {
        Mat tile = tiles[i].second.clone();
        if (tile.channels() == 1) cvtColor(tile, tile, COLOR_GRAY2BGR);
        addLabel(tile, tiles[i].first).copyTo(
            mosaic(Rect(i * W, headerH, W, H + labelH)));
    }
    return mosaic;
}

void tampil(const string& win, const Mat& img,
    const string& outDir, const string& fname) {
    imwrite(outDir + "\\" + fname + ".bmp", img);
    namedWindow(win, WINDOW_NORMAL);
    resizeWindow(win, min(img.cols, 1500), min(img.rows + 30, 750));
    imshow(win, img);
    cout << ">> " << win << " — screenshot lalu tekan ENTER.\n";
    waitKey(0);
    destroyWindow(win);
}

// Konversi ke grayscale, lalu normalisasi ke 8-bit untuk display
Mat toDisplay(const Mat& src) {
    Mat gray, norm;
    if (src.channels() == 3) cvtColor(src, gray, COLOR_BGR2GRAY);
    else gray = src.clone();
    normalize(gray, norm, 0, 255, NORM_MINMAX, CV_8U);
    return norm;
}

// ============================================================
// PERCOBAAN 2: Sobel via filter2D (kernel manual)
// ============================================================
Mat sobelFilter2D(const Mat& src) {
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F);

    Mat kx = (Mat_<float>(3, 3) <<
        -1.f, 0.f, 1.f,
        -2.f, 0.f, 2.f,
        -1.f, 0.f, 1.f);

    Mat ky = (Mat_<float>(3, 3) <<
        -1.f, -2.f, -1.f,
        0.f, 0.f, 0.f,
        1.f, 2.f, 1.f);

    Mat gx, gy, mag;
    filter2D(gray, gx, CV_32F, kx);
    filter2D(gray, gy, CV_32F, ky);

    mag = Mat::zeros(gray.size(), CV_32F);
    for (int y = 0; y < gray.rows; y++)
        for (int x = 0; x < gray.cols; x++)
            mag.at<float>(y, x) = sqrt(
                gx.at<float>(y, x) * gx.at<float>(y, x) +
                gy.at<float>(y, x) * gy.at<float>(y, x));

    Mat result;
    normalize(mag, result, 0, 255, NORM_MINMAX, CV_8U);
    return result;
}

// ============================================================
// PERCOBAAN 3: Prewitt Filter (manual via filter2D)
// ============================================================
Mat prewittFilter(const Mat& src) {
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F);

    Mat kx = (Mat_<float>(3, 3) <<
        -1.f, 0.f, 1.f,
        -1.f, 0.f, 1.f,
        -1.f, 0.f, 1.f);

    Mat ky = (Mat_<float>(3, 3) <<
        -1.f, -1.f, -1.f,
        0.f, 0.f, 0.f,
        1.f, 1.f, 1.f);

    Mat gx, gy, mag;
    filter2D(gray, gx, CV_32F, kx);
    filter2D(gray, gy, CV_32F, ky);

    mag = Mat::zeros(gray.size(), CV_32F);
    for (int y = 0; y < gray.rows; y++)
        for (int x = 0; x < gray.cols; x++)
            mag.at<float>(y, x) = sqrt(
                gx.at<float>(y, x) * gx.at<float>(y, x) +
                gy.at<float>(y, x) * gy.at<float>(y, x));

    Mat result;
    normalize(mag, result, 0, 255, NORM_MINMAX, CV_8U);
    return result;
}

// ============================================================
// MAIN
// ============================================================
int main() {
    FILE* devNull = nullptr;
    freopen_s(&devNull, "nul", "w", stderr);

    string imgPath = "C:\\Users\\riyadh\\source\\repos\\OpenCV_Test\\OpenCV_Test\\Folder Gambar\\picture_p4.bmp";
    Mat original = imread(imgPath, IMREAD_COLOR);

    FILE* con = nullptr;
    freopen_s(&con, "CON", "w", stderr);

    if (original.empty()) {
        cerr << "ERROR: Gagal membuka gambar!\nPath: " << imgPath << "\n";
        system("pause"); return -1;
    }

    Mat src;
    resize(original, src, Size(300, 225));

    string outDir = "C:\\Users\\riyadh\\source\\repos\\OpenCV_Test\\OpenCV_Test\\Folder Gambar\\hasil_edge";
    _mkdir(outDir.c_str());

    cout << "========================================\n";
    cout << "  PRAKTIKUM - EDGE DETECTION\n";
    cout << "========================================\n\n";

    // ════════════════════════════════════════════════════════
    // PERCOBAAN 1: Variasi parameter cv::Sobel()
    // ════════════════════════════════════════════════════════
    cout << "[PERCOBAAN 1] Variasi parameter cv::Sobel()\n";

    Mat e_base, e_a, e_b, e_c;
    Sobel(src, e_base, CV_32F, 1, 1, 3, 1.0, 0.0, BORDER_DEFAULT);
    Sobel(src, e_a, CV_32F, 2, 2, 5, 1.0, 0.0, BORDER_DEFAULT);
    Sobel(src, e_b, CV_32F, 3, 3, 7, 1.0, 0.0, BORDER_DEFAULT);
    Sobel(src, e_c, CV_32F, 2, 2, 11, 1.0, 0.0, BORDER_DEFAULT);

    // FIX: label pendek agar tidak berdempetan
    vector<pair<string, Mat>> p1 = {
        {"Original",      src},
        {"dx=1 dy=1 k=3", toDisplay(e_base)},
        {"dx=2 dy=2 k=5", toDisplay(e_a)},
        {"dx=3 dy=3 k=7", toDisplay(e_b)},
        {"dx=2 dy=2 k=11",toDisplay(e_c)},
    };

    imwrite(outDir + "\\p1_sobel_base.bmp", toDisplay(e_base));
    imwrite(outDir + "\\p1_sobel_a.bmp", toDisplay(e_a));
    imwrite(outDir + "\\p1_sobel_b.bmp", toDisplay(e_b));
    imwrite(outDir + "\\p1_sobel_c.bmp", toDisplay(e_c));

    tampil("Percobaan 1: Variasi Parameter Sobel",
        makeMosaic(p1, "PERCOBAAN 1: cv::Sobel() - Variasi dx, dy, ksize"),
        outDir, "mosaic_p1");

    // ════════════════════════════════════════════════════════
    // PERCOBAAN 2: Sobel via filter2D
    // ════════════════════════════════════════════════════════
    cout << "[PERCOBAAN 2] Sobel via filter2D\n";

    Mat sobelF2D = sobelFilter2D(src);

    Mat sx, sy, sobelMag, sobelCV;
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    Sobel(gray, sx, CV_32F, 1, 0, 3);
    Sobel(gray, sy, CV_32F, 0, 1, 3);
    magnitude(sx, sy, sobelMag);
    normalize(sobelMag, sobelCV, 0, 255, NORM_MINMAX, CV_8U);

    vector<pair<string, Mat>> p2 = {
        {"Original",          src},
        {"Sobel cv::Sobel()", sobelCV},
        {"Sobel filter2D()",  sobelF2D},
    };

    imwrite(outDir + "\\p2_sobel_cv.bmp", sobelCV);
    imwrite(outDir + "\\p2_sobel_f2d.bmp", sobelF2D);

    tampil("Percobaan 2: Sobel via filter2D",
        makeMosaic(p2, "PERCOBAAN 2: Sobel cv::Sobel() vs filter2D() Manual Kernel"),
        outDir, "mosaic_p2");

    // ════════════════════════════════════════════════════════
    // PERCOBAAN 3: Prewitt Filter
    // ════════════════════════════════════════════════════════
    cout << "[PERCOBAAN 3] Prewitt Filter\n";

    Mat prewitt = prewittFilter(src);

    Mat canny;
    Canny(gray, canny, 50, 150);

    vector<pair<string, Mat>> p3 = {
        {"Original",       src},
        {"Sobel filter2D", sobelF2D},
        {"Prewitt",        prewitt},
        {"Canny (ref)",    canny},
    };

    imwrite(outDir + "\\p3_prewitt.bmp", prewitt);

    tampil("Percobaan 3: Prewitt Filter",
        makeMosaic(p3, "PERCOBAAN 3: Prewitt Filter vs Sobel vs Canny"),
        outDir, "mosaic_p3");

    // ════════════════════════════════════════════════════════
    // RINGKASAN: Gx, Gy, dan Magnitude
    // ════════════════════════════════════════════════════════
    cout << "[RINGKASAN] Komponen Gx, Gy, dan Magnitude\n";

    Mat gx_sobel, gy_sobel, gx_prev, gy_prev;
    Mat grayF;
    gray.convertTo(grayF, CV_32F);

    // FIX E0312: deklarasi kernel sebagai Mat terpisah
    Mat ks_x = (Mat_<float>(3, 3) << -1.f, 0.f, 1.f, -2.f, 0.f, 2.f, -1.f, 0.f, 1.f);
    Mat ks_y = (Mat_<float>(3, 3) << -1.f, -2.f, -1.f, 0.f, 0.f, 0.f, 1.f, 2.f, 1.f);
    Mat kp_x = (Mat_<float>(3, 3) << -1.f, 0.f, 1.f, -1.f, 0.f, 1.f, -1.f, 0.f, 1.f);
    Mat kp_y = (Mat_<float>(3, 3) << -1.f, -1.f, -1.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f);

    filter2D(grayF, gx_sobel, CV_32F, ks_x);
    filter2D(grayF, gy_sobel, CV_32F, ks_y);
    filter2D(grayF, gx_prev, CV_32F, kp_x);
    filter2D(grayF, gy_prev, CV_32F, kp_y);

    vector<pair<string, Mat>> summary = {
        {"Original",      src},
        {"Sobel Gx",      toDisplay(gx_sobel)},
        {"Sobel Gy",      toDisplay(gy_sobel)},
        {"Sobel Mag",     sobelF2D},
        {"Prewitt Gx",    toDisplay(gx_prev)},
        {"Prewitt Gy",    toDisplay(gy_prev)},
        {"Prewitt Mag",   prewitt},
    };

    imwrite(outDir + "\\ringkasan_sobel_gx.bmp", toDisplay(gx_sobel));
    imwrite(outDir + "\\ringkasan_sobel_gy.bmp", toDisplay(gy_sobel));
    imwrite(outDir + "\\ringkasan_prewitt_gx.bmp", toDisplay(gx_prev));
    imwrite(outDir + "\\ringkasan_prewitt_gy.bmp", toDisplay(gy_prev));

    tampil("Ringkasan: Gx, Gy, Magnitude",
        makeMosaic(summary, "RINGKASAN: Sobel & Prewitt - Gx, Gy, dan Magnitude"),
        outDir, "mosaic_ringkasan");

    cout << "\n========================================\n";
    cout << "Semua hasil disimpan di:\n" << outDir << "\n";
    cout << "========================================\n";
    system("pause");
    return 0;
}
