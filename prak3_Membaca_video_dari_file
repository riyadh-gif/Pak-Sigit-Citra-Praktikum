#include <opencv2/opencv.hpp>
#include <iostream>
#include <windows.h>

using namespace std;
using namespace cv;

int main() {
    HANDLE hStderr = GetStdHandle(STD_ERROR_HANDLE);
    HANDLE hFile = CreateFileA("nul", GENERIC_WRITE, FILE_SHARE_WRITE,
        NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile != INVALID_HANDLE_VALUE)
        SetStdHandle(STD_ERROR_HANDLE, hFile);

    string video_path = "chaplin.mp4";
    VideoCapture cap(video_path);

    SetStdHandle(STD_ERROR_HANDLE, hStderr);
    if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);

    if (!cap.isOpened()) {
        cerr << "ERROR: Tidak bisa membuka file video!" << endl;
        cerr << "Pastikan file '" << video_path << "' ada di folder ini" << endl;
        system("pause");
        return -1;
    }

    int frame_width = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);
    int total_frames = (int)cap.get(CAP_PROP_FRAME_COUNT);

    cout << "\n=====================================\n";
    cout << "     VIDEO PLAYER OpenCV\n";
    cout << "=====================================\n";
    cout << "File: " << video_path << "\n";
    cout << "Resolusi: " << frame_width << " x " << frame_height << "\n";
    cout << "FPS: " << fps << "\n";
    cout << "Total Frames: " << total_frames << "\n";
    cout << "Durasi: " << (total_frames / fps) << " detik\n";
    cout << "=====================================\n";
    cout << "\n[KONTROL KEYBOARD]\n";
    cout << "SPACE  = Pause/Resume\n";
    cout << "ESC    = Keluar\n";
    cout << "+/-    = Ubah kecepatan\n";
    cout << "R      = Ulang dari awal\n";
    cout << "=====================================\n";
    cout << "\nTekan SPACE untuk mulai...\n" << endl;

    Mat frame;
    bool paused = true;
    int delay = (int)(1000.0 / fps);
    int current_frame = 0;

    namedWindow("Video Player", WINDOW_NORMAL);
    resizeWindow("Video Player", frame_width, frame_height);

    // ✅ FIX 1: Baca frame pertama SEBELUM loop agar tidak langsung break
    cap >> frame;
    if (frame.empty()) {
        cerr << "ERROR: Tidak bisa membaca frame pertama!" << endl;
        system("pause");
        return -1;
    }
    current_frame = 1;

    while (true) {
        // ✅ FIX 2: Hanya baca frame baru jika tidak pause
        //    dan HANYA setelah frame sebelumnya sudah ditampilkan
        if (!paused) {
            Mat next_frame;
            cap >> next_frame;

            if (next_frame.empty()) {
                cout << "\n[INFO] Video selesai!" << endl;
                paused = true; // Pause di frame terakhir, tidak langsung keluar
            }
            else {
                frame = next_frame;
                current_frame++;
            }
        }

        // Buat salinan untuk overlay teks
        Mat display = frame.clone();

        string frame_info = "Frame: " + to_string(current_frame) + "/" + to_string(total_frames);
        putText(display, frame_info, Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);

        string status = paused ? "PAUSED" : "PLAYING";
        Scalar status_color = paused ? Scalar(0, 0, 255) : Scalar(0, 255, 0);
        putText(display, status, Point(10, 70),
            FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2);

        string speed = "Speed: " + to_string((int)(1000.0 / delay)) + " FPS";
        putText(display, speed, Point(10, 110),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 0), 2);

        putText(display, "SPACE: Play/Pause | ESC: Keluar | R: Ulang | +/-: Kecepatan",
            Point(10, frame_height - 20),
            FONT_HERSHEY_SIMPLEX, 0.45, Scalar(200, 200, 200), 1);

        imshow("Video Player", display);

        // ✅ FIX 3: Saat pause tunggu lebih lama (0 = tunggu selamanya sampai ada key)
        //    Ini mencegah CPU busy-loop dan keypress palsu
        int key = waitKey(paused ? 0 : delay);

        if (key == 27) {  // ESC
            cout << "\n[INFO] Program ditutup oleh user" << endl;
            break;
        }
        else if (key == 32) {  // ✅ FIX 4: Gunakan ASCII 32 bukan ' ' untuk SPACE
            paused = !paused;
            cout << "[ACTION] " << (paused ? "PAUSED" : "RESUME") << endl;
        }
        else if (key == '+' || key == '=') {
            delay = max(10, delay - 50);
            cout << "[SPEED] " << (int)(1000.0 / delay) << " FPS" << endl;
        }
        else if (key == '-' || key == '_') {
            delay += 50;
            cout << "[SPEED] " << (int)(1000.0 / delay) << " FPS" << endl;
        }
        else if (key == 'r' || key == 'R') {
            cap.set(CAP_PROP_POS_FRAMES, 0);
            cap >> frame;
            current_frame = 1;
            paused = false;
            cout << "[ACTION] Video diulang dari awal" << endl;
        }
    }

    cout << "\n[INFO] Menutup program..." << endl;
    cap.release();
    destroyAllWindows();

    cout << "[INFO] Program selesai" << endl;
    system("pause");

    return 0;
}
