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
        FONT_HERSHEY_SIMPLEX, 0.82, Scalar(255, 200, 0), 2);

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
// PERCOBAAN 3: Box blur MANUAL (perkalian matriks 3x3)
// ============================================================
Mat boxBlurManual(const Mat& src) {
    Mat dst = src.clone();
    // Kernel 3x3 rata (averaging)
    float kernel[3][3] = {
        {1 / 9.f, 1 / 9.f, 1 / 9.f},
        {1 / 9.f, 1 / 9.f, 1 / 9.f},
        {1 / 9.f, 1 / 9.f, 1 / 9.f}
    };
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            for (int c = 0; c < src.channels(); c++) {
                float sum = 0;
                for (int ky = -1; ky <= 1; ky++)
                    for (int kx = -1; kx <= 1; kx++)
                        sum += kernel[ky + 1][kx + 1] *
                        src.at<Vec3b>(y + ky, x + kx)[c];
                dst.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(sum);
            }
        }
    }
    return dst;
}

// ============================================================
// PERCOBAAN 6: Kuwahara Filter
// Prinsip: bagi area sekitar piksel jadi 4 kuadran,
// hitung mean & variance tiap kuadran,
// ambil mean dari kuadran dengan variance terkecil
// ============================================================
Mat kuwaharaFilter(const Mat& src, int radius = 5) {
    Mat dst = src.clone();
    int r = radius;

    for (int y = r; y < src.rows - r; y++) {
        for (int x = r; x < src.cols - r; x++) {
            // 4 kuadran: TL, TR, BL, BR (overlap di tengah)
            // masing-masing ukuran (r+1) x (r+1)
            double bestMean[3] = { 0,0,0 };
            double minVar = 1e18;

            // Definisi 4 kuadran
            Rect quads[4] = {
                Rect(x - r, y - r, r + 1, r + 1),  // TL
                Rect(x,   y - r, r + 1, r + 1),  // TR
                Rect(x - r, y,   r + 1, r + 1),  // BL
                Rect(x,   y,   r + 1, r + 1)   // BR
            };

            for (int q = 0; q < 4; q++) {
                // Hitung mean & variance per channel
                double mean[3] = { 0,0,0 };
                double var[3] = { 0,0,0 };
                int n = (r + 1) * (r + 1);

                for (int qy = quads[q].y; qy < quads[q].y + quads[q].height; qy++)
                    for (int qx = quads[q].x; qx < quads[q].x + quads[q].width; qx++)
                        for (int c = 0; c < 3; c++)
                            mean[c] += src.at<Vec3b>(qy, qx)[c];

                for (int c = 0; c < 3; c++) mean[c] /= n;

                for (int qy = quads[q].y; qy < quads[q].y + quads[q].height; qy++)
                    for (int qx = quads[q].x; qx < quads[q].x + quads[q].width; qx++)
                        for (int c = 0; c < 3; c++) {
                            double d = src.at<Vec3b>(qy, qx)[c] - mean[c];
                            var[c] += d * d;
                        }

                double totalVar = (var[0] + var[1] + var[2]) / (3.0 * n);
                if (totalVar < minVar) {
                    minVar = totalVar;
                    for (int c = 0; c < 3; c++) bestMean[c] = mean[c];
                }
            }

            for (int c = 0; c < 3; c++)
                dst.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(bestMean[c]);
        }
    }
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

    string outDir = "C:\\Users\\riyadh\\source\\repos\\OpenCV_Test\\OpenCV_Test\\Folder Gambar\\hasil_smoothing";
    _mkdir(outDir.c_str());

    cout << "========================================\n";
    cout << "  PRAKTIKUM - SMOOTHING FILTER\n";
    cout << "========================================\n\n";

    // ════════════════════════════════════════════════════════
    // PERCOBAAN 1: cv::blur() variasi kernel
    // ════════════════════════════════════════════════════════
    cout << "[PERCOBAAN 1] cv::blur() - variasi kernel\n";
    vector<int> kernels = { 3, 7, 11, 15 };
    vector<pair<string, Mat>> p1 = { {"Original", src} };
    for (int k : kernels) {
        Mat blurred;
        blur(src, blurred, Size(k, k), Point(-1, -1));
        p1.push_back({ "blur " + to_string(k) + "x" + to_string(k), blurred });
        imwrite(outDir + "\\p1_blur_" + to_string(k) + "x" + to_string(k) + ".bmp", blurred);
    }
    tampil("Percobaan 1: cv::blur Variasi Kernel",
        makeMosaic(p1, "PERCOBAAN 1: cv::blur() - Kernel 3x3, 7x7, 11x11, 15x15"),
        outDir, "mosaic_p1");

    // ════════════════════════════════════════════════════════
    // PERCOBAAN 2: filter2D() — box kernel manual via filter2D
    // ════════════════════════════════════════════════════════
    cout << "[PERCOBAAN 2] filter2D() dengan kernel averaging\n";
    vector<pair<string, Mat>> p2 = { {"Original", src} };
    for (int k : kernels) {
        Mat kernel = Mat::ones(k, k, CV_32F) / (float)(k * k);
        Mat result;
        filter2D(src, result, -1, kernel);
        p2.push_back({ "filter2D " + to_string(k) + "x" + to_string(k), result });
        imwrite(outDir + "\\p2_filter2D_" + to_string(k) + ".bmp", result);
    }
    tampil("Percobaan 2: filter2D()",
        makeMosaic(p2, "PERCOBAAN 2: filter2D() - Kernel 3x3, 7x7, 11x11, 15x15"),
        outDir, "mosaic_p2");

    // ════════════════════════════════════════════════════════
    // PERCOBAAN 3: Manual perkalian matriks 3x3
    // ════════════════════════════════════════════════════════
    cout << "[PERCOBAAN 3] Box blur MANUAL (perkalian matriks)\n";
    Mat manualResult = boxBlurManual(src);
    Mat blurCV3;
    blur(src, blurCV3, Size(3, 3));
    vector<pair<string, Mat>> p3 = {
        {"Original",            src},
        {"cv::blur 3x3",        blurCV3},
        {"Manual 3x3 (matriks)",manualResult}
    };
    imwrite(outDir + "\\p3_manual_blur.bmp", manualResult);
    tampil("Percobaan 3: Manual Blur",
        makeMosaic(p3, "PERCOBAAN 3: Manual Perkalian Matriks 3x3 vs cv::blur"),
        outDir, "mosaic_p3");

    // ════════════════════════════════════════════════════════
    // PERCOBAAN 4: GaussianBlur variasi kernel
    // ════════════════════════════════════════════════════════
    cout << "[PERCOBAAN 4] GaussianBlur() - variasi kernel\n";
    vector<pair<string, Mat>> p4 = { {"Original", src} };
    for (int k : kernels) {
        Mat gauss;
        GaussianBlur(src, gauss, Size(k, k), 0);
        p4.push_back({ "Gaussian " + to_string(k) + "x" + to_string(k), gauss });
        imwrite(outDir + "\\p4_gaussian_" + to_string(k) + ".bmp", gauss);
    }
    tampil("Percobaan 4: GaussianBlur",
        makeMosaic(p4, "PERCOBAAN 4: GaussianBlur() - Kernel 3x3, 7x7, 11x11, 15x15"),
        outDir, "mosaic_p4");

    // ════════════════════════════════════════════════════════
    // PERCOBAAN 5: medianBlur variasi kernel
    // ════════════════════════════════════════════════════════
    cout << "[PERCOBAAN 5] medianBlur() - variasi kernel\n";
    vector<pair<string, Mat>> p5 = { {"Original", src} };
    for (int k : kernels) {
        Mat med;
        medianBlur(src, med, k);
        p5.push_back({ "Median " + to_string(k) + "x" + to_string(k), med });
        imwrite(outDir + "\\p5_median_" + to_string(k) + ".bmp", med);
    }
    tampil("Percobaan 5: medianBlur",
        makeMosaic(p5, "PERCOBAAN 5: medianBlur() - Kernel 3x3, 7x7, 11x11, 15x15"),
        outDir, "mosaic_p5");

    // ════════════════════════════════════════════════════════
    // PERCOBAAN 6: Kuwahara Filter
    // ════════════════════════════════════════════════════════
    cout << "[PERCOBAAN 6] Kuwahara Filter (radius 2 & 5)\n";
    cout << "  Proses Kuwahara r=2... (butuh waktu)\n";
    Mat kuw2 = kuwaharaFilter(src, 2);
    cout << "  Proses Kuwahara r=5... (butuh waktu)\n";
    Mat kuw5 = kuwaharaFilter(src, 5);
    Mat gaussRef;
    GaussianBlur(src, gaussRef, Size(11, 11), 0);

    vector<pair<string, Mat>> p6 = {
        {"Original",          src},
        {"Gaussian 11x11",    gaussRef},
        {"Kuwahara r=2",      kuw2},
        {"Kuwahara r=5",      kuw5}
    };
    imwrite(outDir + "\\p6_kuwahara_r2.bmp", kuw2);
    imwrite(outDir + "\\p6_kuwahara_r5.bmp", kuw5);
    tampil("Percobaan 6: Kuwahara Filter",
        makeMosaic(p6, "PERCOBAAN 6: Kuwahara Filter (non-linear, edge-preserving)"),
        outDir, "mosaic_p6");

    cout << "\n========================================\n";
    cout << "Semua hasil disimpan di:\n" << outDir << "\n";
    cout << "========================================\n";
    system("pause");
    return 0;
}
