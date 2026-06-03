// =====================================================================
//  PRAKTIKUM 7 - DOMAIN FREKUENSI (DISCRETE FOURIER TRANSFORM 2-D)
//  Satu program, sekali jalan, semua task kelar.
//
//  Isi (tiap window -> SCREENSHOT -> tekan tombol untuk lanjut):
//   [1] Spektrum TERPUSAT  : low-freq di tengah, high-freq di 4 pojok
//                            (program contoh / pakai cv::dft + fftshift)
//   [2] Spektrum 1 KUADRAN : low-freq di pojok KIRI-ATAS, high di
//                            KANAN-BAWAH  -> SOAL 1
//   [3] DFT MANUAL         : transformasi Fourier 2D dibuat sendiri
//                            dari rumus dasar 1D, dibandingkan dengan
//                            hasil cv::dft  -> SOAL 2
//
//  Rumus dasar (Soal 2):  X[n] = (1/N) * sum_{k=0}^{N-1} x[k]*e^(-j2*pi*k*n/N)
//  2D diperoleh dengan menerapkan DFT 1D ke tiap baris lalu tiap kolom
//  (sifat separable).
//
//  Cara pakai: taruh "lena.bmp" (atau Lenna.png/lena.jpg) di folder yang
//  sama. Kalau tidak ada, dibuat gambar pengganti (sinusoid + kotak) yang
//  spektrumnya justru jelas untuk pembelajaran.
//
//  Compile (Windows / MinGW contoh):
//   g++ praktikum7_frekuensi.cpp -o praktikum7 -std=c++17 \
//       -I"C:/opencv/include" -L"C:/opencv/lib" -lopencv_world4xx
// =====================================================================

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <string>

using namespace cv;
using namespace std;

// Ukuran citra untuk DFT MANUAL. Naik = lebih detail tapi lebih lambat
// (naif O(N^3)). 128 cepat; turunkan ke 64 kalau lemot, naikkan ke 256
// kalau mau spektrum lebih halus.
const int MANUAL_SIZE = 128;

int stepCount = 0;

// ---------------------------------------------------------------------
// Tukar kuadran spektrum supaya DC (frekuensi 0) pindah ke tengah.
void fftshift(Mat& m) {
    m = m(Rect(0, 0, m.cols & -2, m.rows & -2)); // pastikan genap
    int cx = m.cols / 2, cy = m.rows / 2;
    Mat q0(m, Rect(0, 0, cx, cy)), q1(m, Rect(cx, 0, cx, cy));
    Mat q2(m, Rect(0, cy, cx, cy)), q3(m, Rect(cx, cy, cx, cy));
    Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);   // TL <-> BR
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);   // TR <-> BL
}

// Hitung magnitudo spektrum (skala log) dengan cv::dft. BELUM di-shift,
// jadi DC berada di pojok (0,0) / keempat sudut.
Mat dftMagnitude(const Mat& gray) {
    Mat padded;
    int m = getOptimalDFTSize(gray.rows);
    int n = getOptimalDFTSize(gray.cols);
    copyMakeBorder(gray, padded, 0, m - gray.rows, 0, n - gray.cols,
        BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);

    dft(complexI, complexI);

    split(complexI, planes);
    magnitude(planes[0], planes[1], planes[0]);
    Mat magI = planes[0];

    magI += Scalar::all(1);   // skala logaritmik
    log(magI, magI);
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    return magI;
}

// ---------------------------------------------------------------------
// SOAL 2: DFT 1D naif sesuai rumus dasar.
vector<complex<double>> dft1d(const vector<complex<double>>& x) {
    int N = (int)x.size();
    vector<complex<double>> X(N);
    for (int nn = 0; nn < N; nn++) {
        complex<double> sum(0.0, 0.0);
        for (int k = 0; k < N; k++) {
            double ang = -2.0 * CV_PI * k * nn / N;
            sum += x[k] * complex<double>(cos(ang), sin(ang));
        }
        X[nn] = sum / (double)N;   // faktor 1/N sesuai rumus
    }
    return X;
}

// DFT 2D manual = DFT 1D per baris, lalu DFT 1D per kolom (separable).
// Mengembalikan magnitudo skala log (CV_32F), belum di-shift.
Mat manualDFT2D(const Mat& grayF) {
    int R = grayF.rows, C = grayF.cols;
    vector<vector<complex<double>>> g(R, vector<complex<double>>(C));
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            g[i][j] = complex<double>(grayF.at<float>(i, j), 0.0);

    // transform tiap baris
    for (int i = 0; i < R; i++) {
        vector<complex<double>> v(C);
        for (int j = 0; j < C; j++) v[j] = g[i][j];
        v = dft1d(v);
        for (int j = 0; j < C; j++) g[i][j] = v[j];
    }
    // transform tiap kolom
    for (int j = 0; j < C; j++) {
        vector<complex<double>> v(R);
        for (int i = 0; i < R; i++) v[i] = g[i][j];
        v = dft1d(v);
        for (int i = 0; i < R; i++) g[i][j] = v[i];
    }

    Mat mag(R, C, CV_32F);
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            mag.at<float>(i, j) = (float)log(1.0 + abs(g[i][j]));
    return mag;
}

// ---------------------------------------------------------------------
// Helper tampilan
Mat f2u(const Mat& f) {              // float [0,1] -> 8-bit
    Mat n, u;
    normalize(f, n, 0, 1.0, NORM_MINMAX);
    n.convertTo(u, CV_8U, 255.0);
    return u;
}

Mat panel(const Mat& img, const string& label, int sz = 280) {
    Mat im;
    if (img.channels() == 1) cvtColor(img, im, COLOR_GRAY2BGR);
    else im = img.clone();
    resize(im, im, Size(sz, sz));
    Mat canvas(sz + 26, sz, CV_8UC3, Scalar(35, 35, 35));
    im.copyTo(canvas(Rect(0, 26, sz, sz)));
    putText(canvas, label, Point(6, 18), FONT_HERSHEY_SIMPLEX, 0.45,
        Scalar(0, 255, 255), 1, LINE_AA);
    return canvas;
}

Mat row(vector<Mat> p) { Mat o; hconcat(p, o); return o; }

void show(const string& tag, const Mat& fig) {
    stepCount++;
    string num = (stepCount < 10 ? "0" : "") + to_string(stepCount);
    cout << "[" << num << "] " << tag << "   (screenshot, lalu tekan tombol untuk lanjut)\n";
    imshow(tag, fig);
    waitKey(0);
    destroyWindow(tag);
}

// ---------------------------------------------------------------------
Mat loadFirstGray(vector<string> names) {
    for (auto& n : names) {
        Mat m = imread(n, IMREAD_GRAYSCALE);
        if (!m.empty()) { cout << "  -> pakai: " << n << "\n"; return m; }
    }
    return Mat();
}

// Gambar pengganti: gabungan dua sinusoid + kotak.
// Sinusoid -> menghasilkan titik terang simetris di spektrum (jelas untuk belajar).
Mat fakeImage() {
    Mat g(256, 256, CV_8UC1);
    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 256; j++) {
            double v = 128 + 60 * sin(2 * CV_PI * 8 * j / 256.0)
                + 40 * sin(2 * CV_PI * 4 * i / 256.0);
            g.at<uchar>(i, j) = saturate_cast<uchar>(v);
        }
    rectangle(g, Point(80, 80), Point(175, 175), Scalar(255), 2);
    return g;
}

// ---------------------------------------------------------------------
int main() {
    cout << "Memuat gambar...\n";
    Mat gray = loadFirstGray({ "lena.bmp", "Lenna.png", "lena.jpg", "lena.png" });
    if (gray.empty()) { cout << "  -> lena tidak ada, pakai gambar pengganti.\n"; gray = fakeImage(); }

    cout << "\n=== MULAI: tiap window -> screenshot -> tekan tombol untuk lanjut ===\n\n";

    // -----------------------------------------------------------------
    // [1] Spektrum TERPUSAT (program contoh): low di tengah, high di pojok
    {
        Mat magI = dftMagnitude(gray);
        fftshift(magI);
        show("1_spektrum_terpusat",
            row({ panel(gray, "input"),
                  panel(f2u(magI), "spektrum (low di TENGAH)") }));
    }

    // -----------------------------------------------------------------
    // [2] SOAL 1: satu kuadran -> low di KIRI-ATAS, high di KANAN-BAWAH.
    //     Caranya: TIDAK di-shift, lalu ambil kuadran kiri-atas saja.
    //     Pada spektrum tak-shift, DC (0,0) ada di pojok kiri-atas, dan
    //     frekuensi naik menuju tengah (pojok kanan-bawah kuadran ini).
    {
        Mat magI = dftMagnitude(gray);             // tanpa fftshift
        Mat q0 = magI(Rect(0, 0, magI.cols / 2, magI.rows / 2)).clone();
        show("2_spektrum_1kuadran",
            row({ panel(gray, "input"),
                  panel(f2u(q0), "1 kuadran (low KIRI-ATAS)") }));
    }

    // -----------------------------------------------------------------
    // [3] SOAL 2: DFT 2D MANUAL dari rumus dasar, dibandingkan cv::dft.
    {
        Mat small;
        resize(gray, small, Size(MANUAL_SIZE, MANUAL_SIZE));
        Mat smallF; small.convertTo(smallF, CV_32F);

        // versi manual
        Mat magMan = manualDFT2D(smallF);
        fftshift(magMan);

        // versi OpenCV pada citra kecil yang sama (untuk pembanding)
        Mat magCv = dftMagnitude(small);
        fftshift(magCv);

        show("3_dft_manual_vs_opencv",
            row({ panel(small,        "input " + to_string(MANUAL_SIZE) + "px"),
                  panel(f2u(magMan),  "DFT manual (rumus 1D)"),
                  panel(f2u(magCv),   "cv::dft (pembanding)") }));
    }

    destroyAllWindows();
    cout << "\n=== SELESAI. Total " << stepCount << " figur ditampilkan. ===\n";
    return 0;
}
