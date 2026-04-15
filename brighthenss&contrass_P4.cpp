#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <direct.h>      // FIX: ganti filesystem -> _mkdir (no C++17 needed)

using namespace std;
using namespace cv;

// ─── Terapkan transformasi linear ────────────────────────────────────────────
Mat applyTransform(const Mat& src, double alpha, int beta) {
    Mat dst = Mat::zeros(src.size(), src.type());
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++)
            for (int c = 0; c < src.channels(); c++)
                dst.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(
                    alpha * src.at<Vec3b>(y, x)[c] + beta
                    );
    return dst;
}

// ─── Label + border putih di bawah gambar ────────────────────────────────────
Mat addLabel(const Mat& img, const string& label) {
    // Tambah strip label 40px di BAWAH gambar
    Mat out;
    copyMakeBorder(img, out, 0, 40, 0, 0, BORDER_CONSTANT, Scalar(15, 15, 15));
    rectangle(out, Point(0, img.rows), Point(out.cols, out.rows), Scalar(15, 15, 15), FILLED);
    putText(out, label, Point(6, img.rows + 26),
        FONT_HERSHEY_SIMPLEX, 0.52, Scalar(0, 220, 255), 1);
    return out;
}

// ─── Buat grid mosaic ─────────────────────────────────────────────────────────
Mat makeMosaic(const vector<Mat>& images, int cols, const string& judul) {
    int n = (int)images.size();
    int rows = (n + cols - 1) / cols;
    int W = images[0].cols;
    int H = images[0].rows;
    int headerH = 50;

    Mat mosaic(H * rows + headerH, W * cols, images[0].type(), Scalar(30, 30, 30));

    // Header judul
    rectangle(mosaic, Point(0, 0), Point(mosaic.cols, headerH), Scalar(20, 20, 20), FILLED);
    putText(mosaic, judul, Point(12, 34),
        FONT_HERSHEY_SIMPLEX, 0.85, Scalar(255, 200, 0), 2);

    for (int i = 0; i < n; i++) {
        int r = i / cols, c = i % cols;
        images[i].copyTo(mosaic(Rect(c * W, headerH + r * H, W, H)));
    }
    return mosaic;
}

int main() {
    // ── Load gambar ──────────────────────────────────────────────────────────
    string imgPath = "C:\\Users\\riyadh\\source\\repos\\OpenCV_Test\\OpenCV_Test\\Folder Gambar\\picture_p4.bmp";
    Mat original = imread(imgPath);

    if (original.empty()) {
        cerr << "ERROR: Gagal membuka gambar!\nPath: " << imgPath << "\n";
        system("pause");
        return -1;
    }

    Mat src;
    resize(original, src, Size(300, 225));  // ukuran per tile

    // ── Buat folder output (FIX: pakai _mkdir) ───────────────────────────────
    string outDir = "C:\\Users\\riyadh\\source\\repos\\OpenCV_Test\\OpenCV_Test\\Folder Gambar\\hasil_praktikum4";
    _mkdir(outDir.c_str());

    cout << "========================================\n";
    cout << "  PRAKTIKUM 4 - AUTO LINEAR TRANSFORM\n";
    cout << "========================================\n\n";

    // ════════════════════════════════════════════════════════════════════════
    // PERCOBAAN 1: Variasi Alpha (beta = 0)
    // ════════════════════════════════════════════════════════════════════════
    cout << "[PERCOBAAN 1] Variasi Alpha (beta = 0)\n";
    vector<double> alphas = { 1.0, 1.5, 2.0, 3.0 };
    vector<Mat> p1;
    p1.push_back(addLabel(src, "ORIGINAL"));

    for (double a : alphas) {
        Mat res = applyTransform(src, a, 0);
        string lbl = "alpha=" + to_string(a).substr(0, 3) + "  beta=0";
        p1.push_back(addLabel(res, lbl));
        string f = outDir + "\\p1_alpha" + to_string(a).substr(0, 3) + ".bmp";
        imwrite(f, res);
        cout << "  Saved: " << f << "\n";
    }

    Mat mosaic1 = makeMosaic(p1, 5, "PERCOBAAN 1: Variasi Alpha (beta=0)");
    imwrite(outDir + "\\mosaic_percobaan1.bmp", mosaic1);
    cout << "  Mosaic saved.\n\n";

    // ════════════════════════════════════════════════════════════════════════
    // PERCOBAAN 2: Variasi Beta (alpha = 1.0)
    // ════════════════════════════════════════════════════════════════════════
    cout << "[PERCOBAAN 2] Variasi Beta (alpha = 1.0)\n";
    vector<int> betas = { 0, 50, 100 };
    vector<Mat> p2;
    p2.push_back(addLabel(src, "ORIGINAL"));

    for (int b : betas) {
        Mat res = applyTransform(src, 1.0, b);
        string lbl = "alpha=1.0  beta=" + to_string(b);
        p2.push_back(addLabel(res, lbl));
        string f = outDir + "\\p2_beta" + to_string(b) + ".bmp";
        imwrite(f, res);
        cout << "  Saved: " << f << "\n";
    }

    Mat mosaic2 = makeMosaic(p2, 4, "PERCOBAAN 2: Variasi Beta (alpha=1.0)");
    imwrite(outDir + "\\mosaic_percobaan2.bmp", mosaic2);
    cout << "  Mosaic saved.\n\n";

    // ════════════════════════════════════════════════════════════════════════
    // PERCOBAAN 3: Kombinasi Alpha + Beta
    // ════════════════════════════════════════════════════════════════════════
    cout << "[PERCOBAAN 3] Kombinasi Alpha + Beta\n";
    struct Combo { double alpha; int beta; string desc; };
    vector<Combo> combos = {
        { 1.5,  30,  "a=1.5 b=30 [natural]"   },
        { 2.0, -50,  "a=2.0 b=-50 [cinematic]" },
        { 0.5, 100,  "a=0.5 b=100 [washed]"    },
        { 1.2,  20,  "a=1.2 b=20 [subtle]"     },
        { 3.0, -80,  "a=3.0 b=-80 [dramatic]"  },
        { 0.8,  60,  "a=0.8 b=60 [faded]"      },
    };

    vector<Mat> p3;
    p3.push_back(addLabel(src, "ORIGINAL"));

    for (auto& c : combos) {
        Mat res = applyTransform(src, c.alpha, c.beta);
        p3.push_back(addLabel(res, c.desc));
        string safe = c.desc;
        for (char& ch : safe) if (ch == ' ' || ch == '.' || ch == '=' || ch == '[' || ch == ']') ch = '_';
        string f = outDir + "\\p3_" + safe + ".bmp";
        imwrite(f, res);
        cout << "  Saved: " << f << "\n";
    }

    Mat mosaic3 = makeMosaic(p3, 4, "PERCOBAAN 3: Kombinasi Alpha + Beta");
    imwrite(outDir + "\\mosaic_percobaan3.bmp", mosaic3);
    cout << "  Mosaic saved.\n\n";

    // ════════════════════════════════════════════════════════════════════════
    // TAMPILKAN LANGSUNG DI LAYAR — tiap window muncul satu per satu
    // ════════════════════════════════════════════════════════════════════════
    cout << "========================================\n";
    cout << "Menampilkan hasil...\n";
    cout << "[ENTER] untuk lanjut ke percobaan berikutnya\n";
    cout << "========================================\n\n";

    // --- Percobaan 1 ---
    namedWindow("Percobaan 1 - Variasi Alpha", WINDOW_NORMAL);
    resizeWindow("Percobaan 1 - Variasi Alpha", min(mosaic1.cols, 1400), min(mosaic1.rows + 50, 700));
    imshow("Percobaan 1 - Variasi Alpha", mosaic1);
    cout << ">> Percobaan 1 ditampilkan. Screenshot sekarang, lalu tekan ENTER.\n";
    waitKey(0);

    // --- Percobaan 2 ---
    namedWindow("Percobaan 2 - Variasi Beta", WINDOW_NORMAL);
    resizeWindow("Percobaan 2 - Variasi Beta", min(mosaic2.cols, 1400), min(mosaic2.rows + 50, 700));
    imshow("Percobaan 2 - Variasi Beta", mosaic2);
    cout << ">> Percobaan 2 ditampilkan. Screenshot sekarang, lalu tekan ENTER.\n";
    waitKey(0);

    // --- Percobaan 3 ---
    namedWindow("Percobaan 3 - Kombinasi", WINDOW_NORMAL);
    resizeWindow("Percobaan 3 - Kombinasi", min(mosaic3.cols, 1400), min(mosaic3.rows + 50, 800));
    imshow("Percobaan 3 - Kombinasi", mosaic3);
    cout << ">> Percobaan 3 ditampilkan. Screenshot sekarang, lalu tekan ENTER untuk keluar.\n";
    waitKey(0);

    destroyAllWindows();

    cout << "\n========================================\n";
    cout << "Semua file tersimpan di:\n" << outDir << "\n";
    cout << "========================================\n";
    system("pause");
    return 0;
}
