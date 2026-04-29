#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void showEroded(const string& windowName, Mat& src, int morphType, int kernelSize, int iterations = 1) {
    Mat element = getStructuringElement(morphType,
        Size(kernelSize, kernelSize),
        Point(kernelSize / 2, kernelSize / 2));

    Mat result = src.clone();
    for (int i = 0; i < iterations; i++) {
        erode(result, result, element);
    }

    namedWindow(windowName, WINDOW_AUTOSIZE);
    imshow(windowName, result);
}

int main()
{
    // ============================================================
    // Load lena.png untuk percobaan 1-3
    // ============================================================
    Mat lena = imread("lenna.png", IMREAD_COLOR);
    if (lena.empty()) {
        cout << "lena.png tidak ditemukan!" << endl;
        return -1;
    }

    imshow("Original - lena.png", lena);
    cout << "=== PERCOBAAN EROSI - lena.png ===" << endl;
    cout << "Tekan Enter untuk lanjut ke setiap percobaan...\n" << endl;

    // ============================================================
    // PERCOBAAN 1 - Variasi ukuran structuring element
    // ============================================================
    cout << "[1/6] Erosi 3x3 (MORPH_CROSS) - lena" << endl;
    waitKey(0); destroyAllWindows();
    showEroded("Erosi 3x3 - lena", lena, MORPH_CROSS, 3);

    cout << "[2/6] Erosi 7x7 (MORPH_CROSS) - lena" << endl;
    waitKey(0); destroyAllWindows();
    showEroded("Erosi 7x7 - lena", lena, MORPH_CROSS, 7);

    cout << "[3/6] Erosi 11x11 (MORPH_CROSS) - lena" << endl;
    waitKey(0); destroyAllWindows();
    showEroded("Erosi 11x11 - lena", lena, MORPH_CROSS, 11);

    cout << "[4/6] Erosi 15x15 (MORPH_CROSS) - lena" << endl;
    waitKey(0); destroyAllWindows();
    showEroded("Erosi 15x15 - lena", lena, MORPH_CROSS, 15);

    // ============================================================
    // PERCOBAAN 2 - Variasi tipe structuring element (kernel 7x7)
    // ============================================================
    cout << "[5/6] Erosi MORPH_RECT 7x7 - lena" << endl;
    waitKey(0); destroyAllWindows();
    showEroded("Erosi MORPH_RECT - lena", lena, MORPH_RECT, 7);

    cout << "[6/6] Erosi MORPH_ELLIPSE 7x7 - lena" << endl;
    waitKey(0); destroyAllWindows();
    showEroded("Erosi MORPH_ELLIPSE - lena", lena, MORPH_ELLIPSE, 7);

    // ============================================================
    // PERCOBAAN 3 - Looping iterasi erosi pada lena.png
    // ============================================================
    cout << "\n[Iterasi] Erosi N=1 - lena" << endl;
    waitKey(0); destroyAllWindows();
    showEroded("Erosi N=1 - lena", lena, MORPH_CROSS, 3, 1);

    cout << "[Iterasi] Erosi N=5 - lena" << endl;
    waitKey(0); destroyAllWindows();
    showEroded("Erosi N=5 - lena", lena, MORPH_CROSS, 3, 5);

    cout << "[Iterasi] Erosi N=10 - lena" << endl;
    waitKey(0); destroyAllWindows();
    showEroded("Erosi N=10 - lena", lena, MORPH_CROSS, 3, 10);

    // ============================================================
    // PERCOBAAN 4 - Ulangi percobaan 3 dengan reta.jpg (gambar paint)
    // ============================================================
    waitKey(0); destroyAllWindows();

    Mat reta = imread("reta.jpg", IMREAD_COLOR);
    if (reta.empty()) {
        cout << "reta.jpg tidak ditemukan!" << endl;
        return -1;
    }

    imshow("Original - reta.jpg", reta);
    cout << "\n=== PERCOBAAN 4 - reta.jpg (ulang iterasi) ===" << endl;

    cout << "[Iterasi] Erosi N=1 - reta" << endl;
    waitKey(0); destroyAllWindows();
    showEroded("Erosi N=1 - reta", reta, MORPH_CROSS, 3, 1);

    cout << "[Iterasi] Erosi N=5 - reta" << endl;
    waitKey(0); destroyAllWindows();
    showEroded("Erosi N=5 - reta", reta, MORPH_CROSS, 3, 5);

    cout << "[Iterasi] Erosi N=10 - reta" << endl;
    waitKey(0); destroyAllWindows();
    showEroded("Erosi N=10 - reta", reta, MORPH_CROSS, 3, 10);

    cout << "\nSemua percobaan selesai. Tekan Enter untuk keluar." << endl;
    waitKey(0);
    destroyAllWindows();
    return 0;
}
