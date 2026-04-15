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
    // Shadow hitam dulu biar kontras
    putText(out, label, Point(7, img.rows + 34),
        FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 4);
    // Teks utama putih kuning
    putText(out, label, Point(7, img.rows + 34),
        FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 230, 255), 2);
    return out;
}

// ============================================================
// BLENDING MANUAL piksel per piksel (tanpa addWeighted)
// Rumus komposit: dst(i,j) = alpha * src1(i,j) + (1-alpha) * src2(i,j)
// ============================================================
Mat blendManual(const Mat& src1, const Mat& src2, double alpha) {
    CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
    Mat dst(src1.rows, src1.cols, src1.type());
    double beta = 1.0 - alpha;
    for (int y = 0; y < src1.rows; y++)
        for (int x = 0; x < src1.cols; x++)
            for (int c = 0; c < src1.channels(); c++)
                dst.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(
                    alpha * src1.at<Vec3b>(y, x)[c] +
                    beta * src2.at<Vec3b>(y, x)[c]
                    );
    return dst;
}

int main() {
    // Suppress OpenCV INFO log
    FILE* devNull = nullptr;
    freopen_s(&devNull, "nul", "w", stderr);

    // ── Load gambar ──────────────────────────────────────────
    string base = "C:\\Users\\riyadh\\source\\repos\\OpenCV_Test\\OpenCV_Test\\Folder Gambar\\";
    string path1 = base + "picture_p4.bmp";        // ganti nama jika perlu
    string path2 = base + "picture_p4.bmp";        // ganti dengan gambar kedua

    Mat src1 = imread(path1, IMREAD_COLOR);
    Mat src2 = imread(path2, IMREAD_COLOR);

    FILE* con = nullptr;
    freopen_s(&con, "CON", "w", stderr);

    if (src1.empty()) { cerr << "ERROR: Gagal buka gambar 1: " << path1 << "\n"; system("pause"); return -1; }
    if (src2.empty()) { cerr << "ERROR: Gagal buka gambar 2: " << path2 << "\n"; system("pause"); return -1; }

    // Samakan ukuran src2 ke src1
    resize(src1, src1, Size(400, 300));
    resize(src2, src2, Size(400, 300));

    // Buat folder output
    string outDir = base + "hasil_praktikum6";
    _mkdir(outDir.c_str());

    cout << "========================================\n";
    cout << "  PRAKTIKUM 6 - COMPOSITING & MATTING\n";
    cout << "========================================\n";
    cout << "Rumus: dst = alpha*src1 + (1-alpha)*src2\n\n";

    // ── Variasi alpha otomatis ────────────────────────────────
    vector<double> alphaList = { 0.0, 0.25, 0.5, 0.75, 1.0 };

    // Header mosaic
    int W = 400, H = 300, headerH = 56, labelH = 50;
    int cols = (int)alphaList.size() + 2; // +2 untuk src1 & src2
    Mat mosaic(H + labelH + headerH, W * cols, CV_8UC3, Scalar(30, 30, 30));
    rectangle(mosaic, Point(0, 0), Point(mosaic.cols, headerH), Scalar(20, 20, 20), FILLED);
    putText(mosaic, "COMPOSITING MANUAL: dst = alpha*src1 + (1-alpha)*src2",
        Point(10, 40), FONT_HERSHEY_SIMPLEX, 0.85, Scalar(255, 200, 0), 2);

    auto putTile = [&](int col, const Mat& img, const string& lbl) {
        Mat tile = addLabel(img, lbl);
        tile.copyTo(mosaic(Rect(col * W, headerH, W, H + labelH)));
    };

    // Kolom 0 & 1: gambar asli
    putTile(0, src1, "Gambar 1 (src1)");
    putTile(1, src2, "Gambar 2 (src2)");

    cout << "Memproses blending...\n";
    for (int i = 0; i < (int)alphaList.size(); i++) {
        double a = alphaList[i];

        // --- MANUAL (piksel per piksel) ---
        Mat dstManual = blendManual(src1, src2, a);

        // --- OpenCV addWeighted (untuk verifikasi) ---
        Mat dstCV;
        addWeighted(src1, a, src2, 1.0 - a, 0.0, dstCV);

        // Simpan
        string tag = "alpha" + to_string((int)(a * 100));
        imwrite(outDir + "\\blend_manual_" + tag + ".bmp", dstManual);
        imwrite(outDir + "\\blend_cv_" + tag + ".bmp", dstCV);

        string lbl = "a=" + to_string(a).substr(0, 3) + " (Manual)";
        putTile(2 + i, dstManual, lbl);

        cout << "  alpha=" << a << " -> saved\n";
    }

    // Simpan mosaic
    imwrite(outDir + "\\mosaic_compositing.bmp", mosaic);

    // ── Tampilkan ─────────────────────────────────────────────
    cout << "\n========================================\n";
    cout << "Menampilkan hasil...\n";
    cout << "[ENTER] untuk keluar\n";
    cout << "========================================\n\n";

    namedWindow("Compositing Manual", WINDOW_NORMAL);
    resizeWindow("Compositing Manual", min(W * cols, 1500), H + labelH + headerH + 20);
    imshow("Compositing Manual", mosaic);
    cout << ">> Mosaic ditampilkan. Screenshot sekarang, lalu tekan ENTER.\n";
    waitKey(0);
    destroyAllWindows();

    // ── Tampilkan side-by-side tiap alpha untuk perbandingan ──
    cout << "\nMenunjukkan perbandingan Manual vs OpenCV per alpha...\n";
    for (double a : alphaList) {
        Mat dstManual = blendManual(src1, src2, a);
        Mat dstCV;
        addWeighted(src1, a, src2, 1.0 - a, 0.0, dstCV);

        // Gabung horizontal: Manual | CV
        Mat compare;
        hconcat(vector<Mat>{
            addLabel(dstManual, "Manual  a=" + to_string(a).substr(0, 3)),
                addLabel(dstCV, "addWeighted a=" + to_string(a).substr(0, 3))
        }, compare);

        string win = "Perbandingan alpha=" + to_string(a).substr(0, 3);
        namedWindow(win, WINDOW_NORMAL);
        resizeWindow(win, W * 2, H + labelH + 20);
        imshow(win, compare);
        cout << ">> alpha=" << a << " ditampilkan. Screenshot lalu tekan ENTER.\n";
        waitKey(0);
        destroyWindow(win);
    }

    cout << "\n========================================\n";
    cout << "Semua hasil disimpan di:\n" << outDir << "\n";
    cout << "========================================\n";
    system("pause");
    return 0;
}
