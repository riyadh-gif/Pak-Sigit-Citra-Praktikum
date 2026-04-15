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
        FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 4);
    putText(out, label, Point(7, img.rows + 34),
        FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 230, 255), 2);
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

void tampil(const string& win, const Mat& img, const string& outDir, const string& fname) {
    imwrite(outDir + "\\" + fname + ".bmp", img);
    namedWindow(win, WINDOW_NORMAL);
    resizeWindow(win, min(img.cols, 1500), min(img.rows + 30, 750));
    imshow(win, img);
    cout << ">> " << win << " — screenshot lalu tekan ENTER.\n";
    waitKey(0);
    destroyWindow(win);
}

// ============================================================
// PERCOBAAN 2: GaussianBlur MANUAL (tanpa fungsi OpenCV)
// Kernel Gaussian 3x3 (sigma=1):
//  1 2 1
//  2 4 2  / 16
//  1 2 1
// ============================================================
Mat gaussianBlurManual(const Mat& src) {
    Mat dst = src.clone();
    // Kernel Gaussian 3x3
    float k[3][3] = {
        {1 / 16.f, 2 / 16.f, 1 / 16.f},
        {2 / 16.f, 4 / 16.f, 2 / 16.f},
        {1 / 16.f, 2 / 16.f, 1 / 16.f}
    };
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            for (int c = 0; c < src.channels(); c++) {
                float sum = 0;
                for (int ky = -1; ky <= 1; ky++)
                    for (int kx = -1; kx <= 1; kx++)
                        sum += k[ky + 1][kx + 1] *
                        src.at<Vec3b>(y + ky, x + kx)[c];
                dst.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(sum);
            }
        }
    }
    return dst;
}

// ============================================================
// PERCOBAAN 2: addWeighted MANUAL
// Rumus: dst = alpha*src1 + beta*src2 + gamma
// ============================================================
Mat addWeightedManual(const Mat& src1, double alpha,
    const Mat& src2, double beta,
    double gamma = 0.0) {
    CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
    Mat dst(src1.rows, src1.cols, src1.type());
    for (int y = 0; y < src1.rows; y++)
        for (int x = 0; x < src1.cols; x++)
            for (int c = 0; c < src1.channels(); c++)
                dst.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(
                    alpha * src1.at<Vec3b>(y, x)[c] +
                    beta * src2.at<Vec3b>(y, x)[c] +
                    gamma
                    );
    return dst;
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

    // Buat versi blur dulu sebagai input sharpening (simulasi "blurred.bmp")
    Mat blurred;
    GaussianBlur(src, blurred, Size(5, 5), 0);

    string outDir = "C:\\Users\\riyadh\\source\\repos\\OpenCV_Test\\OpenCV_Test\\Folder Gambar\\hasil_sharpening";
    _mkdir(outDir.c_str());

    cout << "========================================\n";
    cout << "  PRAKTIKUM - SHARPENING\n";
    cout << "========================================\n\n";

    // ════════════════════════════════════════════════════════
    // PERCOBAAN 1: Variasi alpha & beta pada addWeighted
    // Rumus sharpening: sharp = alpha*original + beta*blurred
    // alpha + beta = 1, tapi alpha > 1 dan beta < 0 → sharpen
    // ════════════════════════════════════════════════════════
    cout << "[PERCOBAAN 1] Variasi alpha & beta pada addWeighted\n";

    // Kombinasi alpha + beta = 1
    struct Combo { double alpha; double beta; string label; };
    vector<Combo> combos = {
        { 1.0,  0.0,  "a=1.0 b=0.0\n(original)" },
        { 1.2, -0.2,  "a=1.2 b=-0.2\n(mild)"    },
        { 1.5, -0.5,  "a=1.5 b=-0.5\n(medium)"  },
        { 2.0, -1.0,  "a=2.0 b=-1.0\n(strong)"  },
        { 3.0, -2.0,  "a=3.0 b=-2.0\n(extreme)" },
    };

    vector<pair<string, Mat>> p1;
    p1.push_back({ "Blurred (input)", blurred });

    for (auto& c : combos) {
        Mat sharp;
        // GaussianBlur dengan kernel 1x1 = tidak blur (identitas)
        Mat gaussSmall;
        GaussianBlur(blurred, gaussSmall, Size(1, 1), 0);
        addWeighted(blurred, c.alpha, gaussSmall, c.beta, 0, sharp);

        string lbl = "a=" + to_string(c.alpha).substr(0, 3) +
            " b=" + to_string(c.beta).substr(0, 4);
        p1.push_back({ lbl, sharp });
        string safe = lbl;
        for (char& ch : safe) if (ch == '.' || ch == '-' || ch == ' ') ch = '_';
        imwrite(outDir + "\\p1_" + safe + ".bmp", sharp);
        cout << "  " << lbl << " -> saved\n";
    }

    tampil("Percobaan 1: Variasi Alpha & Beta",
        makeMosaic(p1, "PERCOBAAN 1: Sharpening - Variasi Alpha & Beta (alpha+beta=1)"),
        outDir, "mosaic_p1");

    // ════════════════════════════════════════════════════════
    // PERCOBAAN 1b: Unsharp masking yang lebih representatif
    // sharp = alpha*original - beta*gaussian_blur(original)
    // Gaussian blur yang lebih kuat agar efek sharpening jelas
    // ════════════════════════════════════════════════════════
    cout << "[PERCOBAAN 1b] Unsharp Masking dengan GaussianBlur kuat\n";

    Mat blurForMask;
    GaussianBlur(blurred, blurForMask, Size(9, 9), 0);

    vector<pair<string, Mat>> p1b;
    p1b.push_back({ "Blurred (input)", blurred });

    struct ComboB { double alpha; double beta; };
    vector<ComboB> combosB = {
        {1.0,  0.0},   // tidak sharpen
        {1.2, -0.2},
        {1.5, -0.5},
        {2.0, -1.0},
        {3.0, -2.0},
    };

    for (auto& c : combosB) {
        Mat sharp;
        addWeighted(blurred, c.alpha, blurForMask, c.beta, 0, sharp);
        string lbl = "a=" + to_string(c.alpha).substr(0, 3) +
            " b=" + to_string(c.beta).substr(0, 4);
        p1b.push_back({ lbl, sharp });
        string safe = lbl;
        for (char& ch : safe) if (ch == '.' || ch == '-' || ch == ' ') ch = '_';
        imwrite(outDir + "\\p1b_" + safe + ".bmp", sharp);
    }

    tampil("Percobaan 1b: Unsharp Masking",
        makeMosaic(p1b, "PERCOBAAN 1b: Unsharp Masking - Variasi Alpha & Beta"),
        outDir, "mosaic_p1b");

    // ════════════════════════════════════════════════════════
    // PERCOBAAN 2: GaussianBlur + addWeighted MANUAL
    // Tanpa fungsi GaussianBlur() & addWeighted() dari OpenCV
    // ════════════════════════════════════════════════════════
    cout << "[PERCOBAAN 2] GaussianBlur + addWeighted MANUAL\n";

    // Gaussian manual
    Mat gaussManual = gaussianBlurManual(blurred);

    // addWeighted manual: sharp = 1.5*blurred + (-0.5)*gaussManual
    Mat sharpManual = addWeightedManual(blurred, 1.5, gaussManual, -0.5);

    // Pembanding dengan OpenCV
    Mat gaussCV, sharpCV;
    GaussianBlur(blurred, gaussCV, Size(3, 3), 0);
    addWeighted(blurred, 1.5, gaussCV, -0.5, 0, sharpCV);

    vector<pair<string, Mat>> p2 = {
        {"Blurred (input)",        blurred},
        {"Gauss Manual (3x3)",     gaussManual},
        {"Sharp MANUAL\na=1.5 b=-0.5", sharpManual},
        {"Gauss OpenCV (3x3)",     gaussCV},
        {"Sharp OpenCV\na=1.5 b=-0.5", sharpCV},
    };

    imwrite(outDir + "\\p2_gauss_manual.bmp", gaussManual);
    imwrite(outDir + "\\p2_sharp_manual.bmp", sharpManual);
    imwrite(outDir + "\\p2_sharp_opencv.bmp", sharpCV);

    tampil("Percobaan 2: Manual vs OpenCV",
        makeMosaic(p2, "PERCOBAAN 2: GaussianBlur + addWeighted Manual vs OpenCV"),
        outDir, "mosaic_p2");

    // ════════════════════════════════════════════════════════
    // RINGKASAN: Original vs Blurred vs Best Sharp
    // ════════════════════════════════════════════════════════
    Mat bestSharp;
    addWeighted(blurred, 1.5, blurForMask, -0.5, 0, bestSharp);

    vector<pair<string, Mat>> summary = {
        {"Original",          src},
        {"Blurred (Gaussian 5x5)", blurred},
        {"Sharp Manual",      sharpManual},
        {"Sharp OpenCV",      sharpCV},
        {"Unsharp Mask (best)", bestSharp},
    };

    tampil("Ringkasan Sharpening",
        makeMosaic(summary, "RINGKASAN: Original vs Blurred vs Hasil Sharpening"),
        outDir, "mosaic_ringkasan");

    cout << "\n========================================\n";
    cout << "Semua hasil disimpan di:\n" << outDir << "\n";
    cout << "========================================\n";
    system("pause");
    return 0;
}
