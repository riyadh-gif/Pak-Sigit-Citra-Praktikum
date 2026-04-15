#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>
#include <iostream>
#include <direct.h>

using namespace cv;
using namespace std;

// ============================================================
// Gambar histogram RGB langsung ke bawah frame (menyatu)
// ============================================================
void drawHistogramOverlay(Mat& frame) {
    const int HIST_H = 120;   // tinggi area histogram
    const int histSize = 256;
    int W = frame.cols;
    int H = frame.rows;

    // --- Hitung frekuensi tiap channel secara manual ---
    int freqB[256] = { 0 }, freqG[256] = { 0 }, freqR[256] = { 0 };
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            Vec3b px = frame.at<Vec3b>(y, x);
            freqB[px[0]]++;
            freqG[px[1]]++;
            freqR[px[2]]++;
        }

    // Cari maxFreq global (normalisasi bersama agar skala setara)
    int maxFreq = 1;
    for (int i = 0; i < histSize; i++) {
        if (freqB[i] > maxFreq) maxFreq = freqB[i];
        if (freqG[i] > maxFreq) maxFreq = freqG[i];
        if (freqR[i] > maxFreq) maxFreq = freqR[i];
    }

    // --- Buat strip histogram (background gelap transparan) ---
    Mat strip(HIST_H, W, CV_8UC3, Scalar(15, 15, 15));

    // Grid horizontal tipis
    for (int h = 0; h < HIST_H; h += HIST_H / 4)
        line(strip, Point(0, h), Point(W, h), Scalar(40, 40, 40), 1);

    // Skala bin ke lebar frame
    double binW = (double)W / histSize;

    // Gambar tiap channel
    for (int i = 1; i < histSize; i++) {
        int x0 = cvRound(binW * (i - 1));
        int x1 = cvRound(binW * i);

        int hB0 = cvRound((double)freqB[i - 1] / maxFreq * (HIST_H - 4));
        int hB1 = cvRound((double)freqB[i] / maxFreq * (HIST_H - 4));
        int hG0 = cvRound((double)freqG[i - 1] / maxFreq * (HIST_H - 4));
        int hG1 = cvRound((double)freqG[i] / maxFreq * (HIST_H - 4));
        int hR0 = cvRound((double)freqR[i - 1] / maxFreq * (HIST_H - 4));
        int hR1 = cvRound((double)freqR[i] / maxFreq * (HIST_H - 4));

        // Blue
        line(strip, Point(x0, HIST_H - hB0), Point(x1, HIST_H - hB1),
            Scalar(255, 80, 80), 1);
        // Green
        line(strip, Point(x0, HIST_H - hG0), Point(x1, HIST_H - hG1),
            Scalar(80, 255, 80), 1);
        // Red
        line(strip, Point(x0, HIST_H - hR0), Point(x1, HIST_H - hR1),
            Scalar(80, 80, 255), 1);
    }

    // Label legenda
    putText(strip, "B", Point(6, 16), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 100, 100), 2);
    putText(strip, "G", Point(24, 16), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(100, 255, 100), 2);
    putText(strip, "R", Point(42, 16), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(100, 100, 255), 2);
    putText(strip, "0", Point(2, HIST_H - 4),
        FONT_HERSHEY_SIMPLEX, 0.35, Scalar(150, 150, 150), 1);
    putText(strip, "255", Point(W - 28, HIST_H - 4),
        FONT_HERSHEY_SIMPLEX, 0.35, Scalar(150, 150, 150), 1);

    // --- Tempel strip ke bawah frame (vconcat) ---
    vconcat(frame, strip, frame);

    // Garis pemisah tipis
    line(frame, Point(0, H), Point(W, H), Scalar(80, 80, 80), 1);

    // Label judul histogram di atas strip
    putText(frame, "RGB Histogram",
        Point(W - 145, H + 14),
        FONT_HERSHEY_SIMPLEX, 0.45, Scalar(200, 200, 200), 1);
}

// ============================================================
// MAIN
// ============================================================
int main() {
    FILE* devNull = nullptr;
    freopen_s(&devNull, "nul", "w", stderr);

    VideoCapture cap(0);  // buka kamera default

    FILE* con = nullptr;
    freopen_s(&con, "CON", "w", stderr);

    if (!cap.isOpened()) {
        cerr << "ERROR: Kamera tidak ditemukan!\n";
        system("pause");
        return -1;
    }

    // Buat folder output untuk screenshot
    string outDir = "C:\\Users\\riyadh\\source\\repos\\OpenCV_Test\\OpenCV_Test\\Folder Gambar\\hasil_tugas_histogram";
    _mkdir(outDir.c_str());

    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;

    cout << "========================================\n";
    cout << "  TUGAS - KAMERA LIVE + HISTOGRAM RGB\n";
    cout << "========================================\n";
    cout << "SPACE : Screenshot & simpan frame\n";
    cout << "ESC   : Keluar\n";
    cout << "========================================\n\n";

    string winName = "Live Camera + RGB Histogram";
    namedWindow(winName, WINDOW_NORMAL);

    int screenshotCount = 0;

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cerr << "[WARN] Frame kosong, skip.\n";
            continue;
        }

        // Tambahkan histogram overlay ke bawah frame
        drawHistogramOverlay(frame);

        // Resize window agar proporsional
        resizeWindow(winName, frame.cols, frame.rows);
        imshow(winName, frame);

        int key = waitKey((int)(1000.0 / fps));

        if (key == 27) {  // ESC
            cout << "[INFO] Keluar.\n";
            break;
        }
        else if (key == 32) {  // SPACE - screenshot
            screenshotCount++;
            string fname = outDir + "\\screenshot_" + to_string(screenshotCount) + ".bmp";
            imwrite(fname, frame);
            cout << "[SAVED] " << fname << "\n";
        }
    }

    cap.release();
    destroyAllWindows();

    cout << "\n========================================\n";
    cout << "Total screenshot: " << screenshotCount << "\n";
    cout << "Tersimpan di    : " << outDir << "\n";
    cout << "========================================\n";
    system("pause");
    return 0;
}
