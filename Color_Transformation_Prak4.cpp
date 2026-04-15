#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <direct.h>

using namespace cv;
using namespace std;

// ============================================================
// UTILITAS
// ============================================================
Mat addLabel(const Mat& img, const string& label) {
    Mat out;
    copyMakeBorder(img, out, 0, 36, 0, 0, BORDER_CONSTANT, Scalar(15, 15, 15));
    putText(out, label, Point(6, img.rows + 24),
        FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0, 220, 255), 1);
    return out;
}

void tampilDanSimpan(const string& judul, const vector<pair<string, Mat>>& hasil,
    const string& outDir, const string& prefix) {
    // Simpan individual
    for (auto& h : hasil) {
        string safe = h.first;
        for (char& c : safe) if (c == ' ' || c == '>' || c == '-') c = '_';
        imwrite(outDir + "\\" + prefix + "_" + safe + ".bmp",
            h.second.channels() == 1 ? h.second : h.second);
    }

    // Buat mosaic
    int W = 300, H = 225, headerH = 46;
    int cols = (int)hasil.size();
    Mat mosaic(H + 36 + headerH, W * cols, CV_8UC3, Scalar(30, 30, 30));
    rectangle(mosaic, Point(0, 0), Point(mosaic.cols, headerH), Scalar(20, 20, 20), FILLED);
    putText(mosaic, judul, Point(10, 32), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 200, 0), 2);

    for (int i = 0; i < cols; i++) {
        Mat tile;
        Mat resized;
        resize(hasil[i].second, resized, Size(W, H));
        // Pastikan 3 channel untuk mosaic
        if (resized.channels() == 1)
            cvtColor(resized, tile, COLOR_GRAY2BGR);
        else
            tile = resized.clone();
        addLabel(tile, hasil[i].first).copyTo(
            mosaic(Rect(i * W, headerH, W, H + 36)));
    }

    string mosaicPath = outDir + "\\" + prefix + "_mosaic.bmp";
    imwrite(mosaicPath, mosaic);

    string winName = judul;
    namedWindow(winName, WINDOW_NORMAL);
    resizeWindow(winName, min(W * cols, 1500), H + 36 + headerH + 20);
    imshow(winName, mosaic);
    cout << ">> " << judul << " ditampilkan. Screenshot lalu tekan ENTER.\n";
    waitKey(0);
    destroyWindow(winName);
}

// ============================================================
// PERCOBAAN 1: Grayscale piksel per piksel (MANUAL)
// ============================================================
Mat toGrayManual(const Mat& src) {
    Mat dst(src.rows, src.cols, CV_8UC1);
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++) {
            Vec3b px = src.at<Vec3b>(y, x);
            // Rumus ITU-R BT.601: 0.299R + 0.587G + 0.114B
            // OpenCV BGR: px[0]=B, px[1]=G, px[2]=R
            dst.at<uchar>(y, x) = saturate_cast<uchar>(
                0.114 * px[0] + 0.587 * px[1] + 0.299 * px[2]);
        }
    return dst;
}

// ============================================================
// PERCOBAAN 2: Konversi pakai cvtColor (fungsi OpenCV)
// ============================================================
// Sudah tersedia langsung via cvtColor()

// ============================================================
// PERCOBAAN 3: Konversi piksel per piksel MANUAL (tanpa cvtColor)
// ============================================================

// --- BGR -> XYZ (manual) ---
Mat toXYZManual(const Mat& src) {
    Mat dst(src.rows, src.cols, CV_8UC3);
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++) {
            Vec3b px = src.at<Vec3b>(y, x);
            double B = px[0] / 255.0, G = px[1] / 255.0, R = px[2] / 255.0;
            // Linearisasi (gamma removal) sRGB
            auto lin = [](double v) {
                return v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4);
            };
            double r = lin(R), g = lin(G), b = lin(B);
            // Matrix sRGB -> XYZ D65
            double X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
            double Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
            double Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
            // Scale ke [0,255] (X max~0.95, Y max~1.0, Z max~1.09)
            dst.at<Vec3b>(y, x) = Vec3b(
                saturate_cast<uchar>(Z * 255.0 / 1.0884),
                saturate_cast<uchar>(Y * 255.0),
                saturate_cast<uchar>(X * 255.0 / 0.9505)
            );
        }
    return dst;
}

// --- BGR -> YCrCb (manual) ---
Mat toYCrCbManual(const Mat& src) {
    Mat dst(src.rows, src.cols, CV_8UC3);
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++) {
            Vec3b px = src.at<Vec3b>(y, x);
            double B = px[0], G = px[1], R = px[2];
            double Y = 0.299 * R + 0.587 * G + 0.114 * B;
            double Cr = (R - Y) * 0.713 + 128.0;
            double Cb = (B - Y) * 0.564 + 128.0;
            dst.at<Vec3b>(y, x) = Vec3b(
                saturate_cast<uchar>(Cb),
                saturate_cast<uchar>(Cr),
                saturate_cast<uchar>(Y)
            );
        }
    return dst;
}

// --- BGR -> HSV (manual) ---
Mat toHSVManual(const Mat& src) {
    Mat dst(src.rows, src.cols, CV_8UC3);
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++) {
            Vec3b px = src.at<Vec3b>(y, x);
            double B = px[0] / 255.0, G = px[1] / 255.0, R = px[2] / 255.0;
            double cmax = max({ R,G,B }), cmin = min({ R,G,B });
            double delta = cmax - cmin;

            double H = 0, S = 0, V = cmax;
            if (delta > 1e-6) {
                S = delta / cmax;
                if (cmax == R) H = 60.0 * fmod((G - B) / delta, 6.0);
                else if (cmax == G) H = 60.0 * ((B - R) / delta + 2.0);
                else                H = 60.0 * ((R - G) / delta + 4.0);
                if (H < 0) H += 360.0;
            }
            // OpenCV HSV scale: H/2, S*255, V*255
            dst.at<Vec3b>(y, x) = Vec3b(
                saturate_cast<uchar>(H / 2.0),
                saturate_cast<uchar>(S * 255.0),
                saturate_cast<uchar>(V * 255.0)
            );
        }
    return dst;
}

// --- BGR -> Lab (manual, D65 illuminant) ---
Mat toLabManual(const Mat& src) {
    Mat dst(src.rows, src.cols, CV_8UC3);
    // D65 reference white
    const double Xn = 0.95047, Yn = 1.00000, Zn = 1.08883;
    auto f = [](double t) {
        return t > 0.008856 ? cbrt(t) : (7.787 * t + 16.0 / 116.0);
    };
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++) {
            Vec3b px = src.at<Vec3b>(y, x);
            double B = px[0] / 255.0, G = px[1] / 255.0, R = px[2] / 255.0;
            auto lin = [](double v) {
                return v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4);
            };
            double r = lin(R), g = lin(G), b = lin(B);
            double X = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / Xn;
            double Y = (r * 0.2126729 + g * 0.7151522 + b * 0.0721750) / Yn;
            double Z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / Zn;

            double L = 116.0 * f(Y) - 16.0;
            double a = 500.0 * (f(X) - f(Y));
            double bv = 200.0 * (f(Y) - f(Z));

            // OpenCV Lab scale: L*255/100, a+128, b+128
            dst.at<Vec3b>(y, x) = Vec3b(
                saturate_cast<uchar>(bv + 128.0),
                saturate_cast<uchar>(a + 128.0),
                saturate_cast<uchar>(L * 255.0 / 100.0)
            );
        }
    return dst;
}

// ============================================================
// MAIN
// ============================================================
int main() {
    FILE* devNull = nullptr;
    freopen_s(&devNull, "nul", "w", stderr);  // suppress OpenCV INFO log

    string imgPath = "C:\\Users\\riyadh\\source\\repos\\OpenCV_Test\\OpenCV_Test\\Folder Gambar\\picture_p4.bmp";
    Mat original = imread(imgPath, IMREAD_COLOR);
    if (original.empty()) {
        cerr << "ERROR: Gagal membuka gambar!\nPath: " << imgPath << "\n";
        system("pause"); return -1;
    }

    FILE* con = nullptr;
    freopen_s(&con, "CON", "w", stderr);

    Mat src;
    resize(original, src, Size(300, 225));

    string outDir = "C:\\Users\\riyadh\\source\\repos\\OpenCV_Test\\OpenCV_Test\\Folder Gambar\\hasil_praktikum5";
    _mkdir(outDir.c_str());

    cout << "========================================\n";
    cout << "  PRAKTIKUM 5 - KONVERSI RUANG WARNA\n";
    cout << "========================================\n\n";

    // ────────────────────────────────────────────────────────
    // PERCOBAAN 1: Grayscale manual (piksel per piksel)
    // ────────────────────────────────────────────────────────
    cout << "[PERCOBAAN 1] RGB -> Grayscale (Manual piksel per piksel)\n";
    Mat grayCV, grayManual;
    cvtColor(src, grayCV, COLOR_BGR2GRAY);
    grayManual = toGrayManual(src);

    tampilDanSimpan("PERCOBAAN 1: RGB -> Grayscale", {
        {"Original",       src},
        {"Gray (cvtColor)", grayCV},
        {"Gray (Manual)",   grayManual}
        }, outDir, "p1");

    // ────────────────────────────────────────────────────────
    // PERCOBAAN 2: Konversi pakai cvtColor (fungsi OpenCV)
    // ────────────────────────────────────────────────────────
    cout << "[PERCOBAAN 2] Konversi warna via cvtColor OpenCV\n";
    Mat xyz, ycrcb, hsv, lab;
    cvtColor(src, xyz, COLOR_BGR2XYZ);
    cvtColor(src, ycrcb, COLOR_BGR2YCrCb);
    cvtColor(src, hsv, COLOR_BGR2HSV);
    cvtColor(src, lab, COLOR_BGR2Lab);

    tampilDanSimpan("PERCOBAAN 2: Konversi via cvtColor", {
        {"Original", src},
        {"XYZ",      xyz},
        {"YCrCb",    ycrcb},
        {"HSV",      hsv},
        {"Lab",      lab}
        }, outDir, "p2");

    // ────────────────────────────────────────────────────────
    // PERCOBAAN 3: Konversi MANUAL piksel per piksel
    // ────────────────────────────────────────────────────────
    cout << "[PERCOBAAN 3] Konversi warna Manual (piksel per piksel)\n";

    // 3a: XYZ manual
    tampilDanSimpan("PERCOBAAN 3a: RGB -> XYZ (Manual)", {
        {"Original",   src},
        {"XYZ (CV)",   xyz},
        {"XYZ (Manual)", toXYZManual(src)}
        }, outDir, "p3a");

    // 3b: YCrCb manual
    tampilDanSimpan("PERCOBAAN 3b: RGB -> YCrCb (Manual)", {
        {"Original",       src},
        {"YCrCb (CV)",     ycrcb},
        {"YCrCb (Manual)", toYCrCbManual(src)}
        }, outDir, "p3b");

    // 3c: HSV manual
    tampilDanSimpan("PERCOBAAN 3c: RGB -> HSV (Manual)", {
        {"Original",     src},
        {"HSV (CV)",     hsv},
        {"HSV (Manual)", toHSVManual(src)}
        }, outDir, "p3c");

    // 3d: Lab manual
    tampilDanSimpan("PERCOBAAN 3d: RGB -> Lab (Manual)", {
        {"Original",     src},
        {"Lab (CV)",     lab},
        {"Lab (Manual)", toLabManual(src)}
        }, outDir, "p3d");

    cout << "\n========================================\n";
    cout << "Semua hasil disimpan di:\n" << outDir << "\n";
    cout << "========================================\n";
    system("pause");
    return 0;
}
