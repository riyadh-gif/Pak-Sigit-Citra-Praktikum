#define _CRT_SECURE_NO_WARNINGS   // FIX C4996: izinkan fungsi C lama
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdio>
#include <ctime>

using namespace std;
using namespace cv;

int tb_tebal = 2;
int tb_opacity = 50;
int tb_ukuran = 100;

void onTrackbar(int, void*) {}

void gambarObjek(Mat& frame, int tebal, int opacity, int ukuran) {
    int W = frame.cols, H = frame.rows;
    Mat overlay = frame.clone();

    rectangle(overlay, Point(20, 130), Point(20 + ukuran, 130 + ukuran), Scalar(0, 255, 255), tebal);
    circle(overlay, Point(W / 2, H - 80), ukuran / 2, Scalar(0, 0, 255), tebal);
    ellipse(overlay, Point(W - 120, 160), Size(ukuran / 2, ukuran / 3), 45, 0, 360, Scalar(255, 100, 0), tebal);
    line(overlay, Point(W - 20, H - 20), Point(W - ukuran - 20, H - ukuran - 20), Scalar(0, 255, 0), tebal);

    double alpha = opacity / 100.0;
    addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame);

    // FIX C4996: Ganti localtime() -> localtime_s()
    time_t now = time(0);
    struct tm timeInfo;
    localtime_s(&timeInfo, &now);  // versi aman untuk MSVC
    char ts[64];
    strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &timeInfo);

    putText(frame, ts, Point(W - 280, H - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
    putText(frame, "KOTAK", Point(22, 125), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 255, 255), 1);
    putText(frame, "LINGKARAN", Point(W / 2 - 40, H - 55), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 255), 1);
    putText(frame, "ELIPS", Point(W - 160, 140), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(255, 100, 0), 1);
    putText(frame, "GARIS", Point(W - 80, H - 80), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 255, 0), 1);
}

void gambarHUD(Mat& frame, int cur, int total, bool paused,
    int delay, bool isCamera, bool isRec) {
    int W = frame.cols, H = frame.rows;
    Mat hud = frame.clone();
    rectangle(hud, Point(0, 0), Point(W, 120), Scalar(0, 0, 0), FILLED);
    addWeighted(hud, 0.4, frame, 0.6, 0, frame);

    putText(frame, isCamera ? "Sumber: KAMERA" : "Sumber: FILE VIDEO",
        Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(200, 200, 200), 1);
    if (!isCamera)
        putText(frame, "Frame: " + to_string(cur) + "/" + to_string(total),
            Point(10, 45), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0, 255, 0), 1);

    putText(frame, paused ? "PAUSED" : "PLAYING",
        Point(10, 70), FONT_HERSHEY_SIMPLEX, 0.65,
        paused ? Scalar(0, 80, 255) : Scalar(0, 255, 0), 2);
    putText(frame, "Speed: " + to_string((int)(1000.0 / max(delay, 1))) + " FPS",
        Point(10, 95), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255, 255, 0), 1);

    if (isRec) {
        circle(frame, Point(W - 30, 20), 10, Scalar(0, 0, 255), FILLED);
        putText(frame, "REC", Point(W - 60, 25), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0, 0, 255), 2);
    }
    putText(frame, "SPACE:Play/Pause | S:Rekam | R:Ulang | +/-:Speed | ESC:Keluar",
        Point(10, H - 8), FONT_HERSHEY_SIMPLEX, 0.38, Scalar(180, 180, 180), 1);
}

int main() {
    // FIX C4996: Ganti freopen() -> freopen_s()
    FILE* devNull = nullptr;
    freopen_s(&devNull, "nul", "w", stderr);  // Suppress semua INFO log OpenCV

    cout << "========================================\n";
    cout << "  PRAKTIKUM 3 - VIDEO + OBJEK + SIMPAN\n";
    cout << "========================================\n";
    cout << "[1] Baca dari FILE VIDEO (chaplin.mp4)\n";
    cout << "[2] Baca dari KAMERA\n";
    cout << "Pilih (1/2): ";
    int pilihan; cin >> pilihan;

    VideoCapture cap;
    bool isCamera = (pilihan == 2);
    if (isCamera) cap.open(0);
    else          cap.open("chaplin.mp4");

    // Restore stderr setelah inisialisasi
    FILE* con = nullptr;
    freopen_s(&con, "CON", "w", stderr);

    if (!cap.isOpened()) {
        cerr << "ERROR: Gagal membuka sumber video!\n";
        system("pause"); return -1;
    }

    int    W = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int    H = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0 || isCamera) fps = 30.0;
    int total = isCamera ? 0 : (int)cap.get(CAP_PROP_FRAME_COUNT);

    cout << "\nResolusi   : " << W << " x " << H << "\n";
    cout << "FPS        : " << fps << "\n";
    if (!isCamera) cout << "Total Frame: " << total << "\n";
    cout << "========================================\n\n";

    string      out_path = "output_hasil.mp4";
    VideoWriter writer;
    int         fourcc = VideoWriter::fourcc('m', 'p', '4', 'v');
    bool        isRec = false;

    string winName = "Praktikum 3 - Video Player";
    namedWindow(winName, WINDOW_NORMAL);
    resizeWindow(winName, W, H);
    createTrackbar("Tebal Garis", winName, &tb_tebal, 10, onTrackbar);
    createTrackbar("Opacity %", winName, &tb_opacity, 100, onTrackbar);
    createTrackbar("Ukuran Obj", winName, &tb_ukuran, 200, onTrackbar);

    Mat  frame;
    bool paused = !isCamera;
    int  delay = (int)(1000.0 / fps);
    int  cur = 0;

    cap >> frame;
    if (frame.empty()) {
        cerr << "ERROR: Frame pertama kosong!\n";
        system("pause"); return -1;
    }
    if (!isCamera) cur = 1;

    cout << "SPACE:Play/Pause | S:Rekam | R:Ulang | +/-:Speed | ESC:Keluar\n\n";

    while (true) {
        if (!paused) {
            Mat next; cap >> next;
            if (next.empty()) {
                if (!isCamera) { cout << "[INFO] Video selesai!\n"; paused = true; }
            }
            else {
                frame = next;
                if (!isCamera) cur++;
            }
        }

        if (frame.empty()) { waitKey(30); continue; }

        Mat display = frame.clone();
        gambarObjek(display, max(1, tb_tebal), tb_opacity, max(20, tb_ukuran));
        gambarHUD(display, cur, total, paused, delay, isCamera, isRec);
        imshow(winName, display);

        if (isRec) {
            if (!writer.isOpened()) writer.open(out_path, fourcc, fps, Size(W, H));
            writer.write(display);
        }

        int key = waitKey(paused ? 30 : delay);

        if (key == 27) { cout << "[INFO] Keluar.\n"; break; }
        else if (key == 32) { paused = !paused; cout << "[ACTION] " << (paused ? "PAUSE" : "RESUME") << "\n"; }
        else if (key == 's' || key == 'S') {
            isRec = !isRec;
            if (isRec) { writer.open(out_path, fourcc, fps, Size(W, H)); cout << "[REC] Merekam: " << out_path << "\n"; }
            else { writer.release(); cout << "[REC] Disimpan: " << out_path << "\n"; }
        }
        else if ((key == 'r' || key == 'R') && !isCamera) {
            cap.set(CAP_PROP_POS_FRAMES, 0); cap >> frame; cur = 1; paused = false;
            cout << "[ACTION] Diulang dari awal\n";
        }
        else if (key == '+' || key == '=') { delay = max(10, delay - 10); cout << "[SPEED] " << (int)(1000.0 / delay) << " FPS\n"; }
        else if (key == '-' || key == '_') { delay += 10;               cout << "[SPEED] " << (int)(1000.0 / delay) << " FPS\n"; }
    }

    if (writer.isOpened()) { writer.release(); cout << "[INFO] Tersimpan: " << out_path << "\n"; }
    cap.release(); destroyAllWindows();
    cout << "[INFO] Program selesai.\n";
    system("pause");
    return 0;
}
