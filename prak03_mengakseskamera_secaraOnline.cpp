#include <opencv2/opencv.hpp>
#include <iostream>
#include <windows.h>

using namespace std;
using namespace cv;

int main() {
    // ✅ FIX: Redirect stderr ke null device
    // Menyembunyikan semua pesan INFO/WARNING OpenCV (GTK failed, TBB failed, dll)
    HANDLE hStderr = GetStdHandle(STD_ERROR_HANDLE);
    HANDLE hNull = CreateFileA(
        "nul", GENERIC_WRITE, FILE_SHARE_WRITE,
        NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL
    );
    if (hNull != INVALID_HANDLE_VALUE)
        SetStdHandle(STD_ERROR_HANDLE, hNull);

    // Buka kamera (0 = kamera default)
    VideoCapture cap(0);

    // ✅ Restore stderr setelah inisialisasi selesai
    SetStdHandle(STD_ERROR_HANDLE, hStderr);
    if (hNull != INVALID_HANDLE_VALUE)
        CloseHandle(hNull);

    // Cek apakah kamera berhasil dibuka
    if (!cap.isOpened()) {
        cerr << "Error: Tidak bisa membuka kamera!" << endl;
        system("pause");
        return -1;
    }

    cout << "Kamera berhasil dibuka. Tekan ESC untuk keluar." << endl;

    while (true) {
        Mat frame;
        cap >> frame;

        if (frame.empty()) {
            cerr << "Error: Frame kosong!" << endl;
            break;
        }

        imshow("Frame", frame);

        // Tunggu 25ms, keluar jika ESC (ASCII 27)
        if (waitKey(25) == 27)
            break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
