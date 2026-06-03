// =====================================================================
//  PRAKTIKUM 8 - FILTER BANK (WAVELET + GABOR FILTER)
//  Satu program, sekali jalan, semua task kelar.
//
//  Isi (tiap window -> SCREENSHOT -> tekan tombol untuk lanjut):
//   [1] WAVELET   : input | dekomposisi Haar (LL + 3 detail) | rekonstruksi
//                   (membuktikan DWT lalu IDWT mengembalikan citra asli)
//   [2] GABOR KERNEL   : bentuk kernel Gabor pada 4 orientasi (0/45/90/135)
//   [3] GABOR ORIENTASI: respon citra terhadap 4 orientasi -> selektivitas sudut
//   [4] GABOR FREKUENSI: respon citra pada beberapa lambda -> selektivitas frekuensi
//
//  Wavelet = kombinasi low-pass & high-pass filter (Haar).
//  Gabor   = kernel Gaussian dimodulasi gelombang sinus (analisis tekstur
//            terlokalisasi pada orientasi & frekuensi tertentu).
//
//  Cara pakai: taruh "lena.bmp" (atau Lenna.png/lena.jpg) di folder yang
//  sama. Kalau tidak ada, dibuat gambar pengganti (garis/kotak/lingkaran)
//  yang justru bagus untuk demo orientasi Gabor.
//
//  Compile (Windows / MinGW contoh):
//   g++ praktikum8_filterbank.cpp -o praktikum8 -std=c++17 \
//       -I"C:/opencv/include" -L"C:/opencv/lib" -lopencv_world4xx
// =====================================================================

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

using namespace cv;
using namespace std;

const double S = sqrt(2.0);   // faktor normalisasi Haar (0.707 = 1/sqrt2)
int stepCount = 0;

// ---------------------------------------------------------------------
// ============================ WAVELET HAAR ===========================
// Transform Haar 1D sepanjang BARIS (horizontal): kiri = aproksimasi (L),
// kanan = detail (H).
void haarRows(const Mat& in, Mat& out) {
    int R = in.rows, C = in.cols, h = C / 2;
    out = Mat::zeros(R, C, CV_32F);
    for (int i = 0; i < R; i++)
        for (int j = 0; j < h; j++) {
            float a = in.at<float>(i, 2 * j), b = in.at<float>(i, 2 * j + 1);
            out.at<float>(i, j)     = (float)((a + b) / S);   // L
            out.at<float>(i, h + j) = (float)((a - b) / S);   // H
        }
}
// Transform Haar 1D sepanjang KOLOM (vertical): atas = L, bawah = H.
void haarCols(const Mat& in, Mat& out) {
    int R = in.rows, C = in.cols, h = R / 2;
    out = Mat::zeros(R, C, CV_32F);
    for (int j = 0; j < C; j++)
        for (int i = 0; i < h; i++) {
            float a = in.at<float>(2 * i, j), b = in.at<float>(2 * i + 1, j);
            out.at<float>(i, j)     = (float)((a + b) / S);
            out.at<float>(h + i, j) = (float)((a - b) / S);
        }
}
// Inverse
void ihaarCols(const Mat& in, Mat& out) {
    int R = in.rows, C = in.cols, h = R / 2;
    out = Mat::zeros(R, C, CV_32F);
    for (int j = 0; j < C; j++)
        for (int i = 0; i < h; i++) {
            float L = in.at<float>(i, j), H = in.at<float>(h + i, j);
            out.at<float>(2 * i, j)     = (float)((L + H) / S);
            out.at<float>(2 * i + 1, j) = (float)((L - H) / S);
        }
}
void ihaarRows(const Mat& in, Mat& out) {
    int R = in.rows, C = in.cols, h = C / 2;
    out = Mat::zeros(R, C, CV_32F);
    for (int i = 0; i < R; i++)
        for (int j = 0; j < h; j++) {
            float L = in.at<float>(i, j), H = in.at<float>(i, h + j);
            out.at<float>(i, 2 * j)     = (float)((L + H) / S);
            out.at<float>(i, 2 * j + 1) = (float)((L - H) / S);
        }
}

// DWT 1 level: baris dulu, lalu kolom. Hasil: TL=LL, sisanya detail.
Mat dwt(const Mat& img) { Mat t, o; haarRows(img, t); haarCols(t, o); return o; }
// IDWT: kebalikan urutan.
Mat idwt(const Mat& d) { Mat t, o; ihaarCols(d, t); ihaarRows(t, o); return o; }

// Susun citra dekomposisi yang enak dilihat:
// LL dinormalisasi sendiri (tampak seperti citra kecil), tiap detail
// di-abs lalu dinormalisasi sendiri supaya tepi-tepinya kelihatan.
Mat decompDisplay(const Mat& d) {
    int R = d.rows, C = d.cols, hr = R / 2, hc = C / 2;
    Mat disp(R, C, CV_8U, Scalar(0));
    auto place = [&](Rect r, bool isLL) {
        Mat band = d(r).clone();
        if (!isLL) band = abs(band);
        Mat n; normalize(band, n, 0, 255, NORM_MINMAX);
        Mat u; n.convertTo(u, CV_8U);
        u.copyTo(disp(r));
    };
    place(Rect(0, 0, hc, hr), true);    // LL (aproksimasi)
    place(Rect(hc, 0, hc, hr), false);  // detail
    place(Rect(0, hr, hc, hr), false);  // detail
    place(Rect(hc, hr, hc, hr), false); // detail
    return disp;
}

// ---------------------------------------------------------------------
// ============================ GABOR FILTER ===========================
// Kernel Gabor: Gaussian dikalikan gelombang kosinus berarah.
Mat gaborKernel(int ks, double sigma, double thetaDeg, double lambda,
                double psiDeg, double gamma) {
    int hks = ks / 2;
    Mat kernel(ks, ks, CV_32F);
    double th = thetaDeg * CV_PI / 180.0;
    double ps = psiDeg   * CV_PI / 180.0;
    for (int y = -hks; y <= hks; y++)
        for (int x = -hks; x <= hks; x++) {
            double xr =  x * cos(th) + y * sin(th);
            double yr = -x * sin(th) + y * cos(th);
            double g = exp(-(xr * xr + gamma * gamma * yr * yr) / (2 * sigma * sigma))
                       * cos(2 * CV_PI * xr / lambda + ps);
            kernel.at<float>(hks + y, hks + x) = (float)g;
        }
    return kernel;
}

// Respon energi Gabor = magnitudo dari pasangan kuadratur (cos & sin),
// sehingga peta orientasi tidak tergantung fase.
Mat gaborResponse(const Mat& srcF, double sigma, double thetaDeg,
                  double lambda, double gamma, int ks = 31) {
    Mat kr = gaborKernel(ks, sigma, thetaDeg, lambda, 0,  gamma);
    Mat ki = gaborKernel(ks, sigma, thetaDeg, lambda, 90, gamma);
    Mat r, im, mag;
    filter2D(srcF, r,  CV_32F, kr);
    filter2D(srcF, im, CV_32F, ki);
    magnitude(r, im, mag);
    return mag;
}

// ---------------------------------------------------------------------
// ===========================  TAMPILAN  ==============================
Mat f2u(const Mat& f) {
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
Mat fakeImage() {
    Mat g(512, 512, CV_8UC1, Scalar(60));
    for (int i = 0; i < 512; i++)
        for (int j = 0; j < 512; j++)
            if (((i + j) / 16) % 2 == 0) g.at<uchar>(i, j) += 40;   // garis diagonal
    rectangle(g, Point(90, 90), Point(240, 240), Scalar(220), -1);
    circle(g, Point(360, 360), 80, Scalar(170), -1);
    for (int j = 0; j < 512; j++) g.at<uchar>(400, j) = 255;        // garis horizontal
    for (int i = 0; i < 512; i++) g.at<uchar>(i, 150) = 255;        // garis vertikal
    return g;
}

// ---------------------------------------------------------------------
int main() {
    cout << "Memuat gambar...\n";
    Mat gray0 = loadFirstGray({"lena.bmp", "Lenna.png", "lena.jpg", "lena.png"});
    if (gray0.empty()) { cout << "  -> lena tidak ada, pakai gambar pengganti.\n"; gray0 = fakeImage(); }

    Mat gray;
    resize(gray0, gray, Size(512, 512));   // genap & seragam untuk DWT + Gabor

    cout << "\n=== MULAI: tiap window -> screenshot -> tekan tombol untuk lanjut ===\n\n";

    // -----------------------------------------------------------------
    // [1] WAVELET: dekomposisi Haar + rekonstruksi
    {
        Mat f; gray.convertTo(f, CV_32F);
        Mat d = dwt(f);                 // dekomposisi 1 level
        Mat r = idwt(d);                // rekonstruksi

        // cek error rekonstruksi (harus mendekati nol)
        double err = norm(f, r, NORM_INF);
        cout << "    -> error maksimum rekonstruksi (IDWT) = " << err << " (idealnya ~0)\n";

        Mat rec8; r.convertTo(rec8, CV_8U);
        show("1_wavelet_haar",
             row({ panel(gray, "input"),
                   panel(decompDisplay(d), "dekomposisi (LL + detail)"),
                   panel(rec8, "rekonstruksi (IDWT)") }));
    }

    // siapkan citra float ternormalisasi untuk Gabor
    Mat srcF; gray.convertTo(srcF, CV_32F, 1.0 / 255.0);
    double sigma = 5, lambda = 10, gamma = 0.5;
    double thetas[4] = {0, 45, 90, 135};

    // -----------------------------------------------------------------
    // [2] GABOR KERNEL pada 4 orientasi
    {
        vector<Mat> p;
        for (double th : thetas)
            p.push_back(panel(f2u(gaborKernel(31, sigma, th, lambda, 0, gamma)),
                              "kernel " + to_string((int)th) + " deg"));
        show("2_gabor_kernel", row(p));
    }

    // -----------------------------------------------------------------
    // [3] GABOR ORIENTASI: respon citra terhadap 4 sudut (lambda tetap)
    {
        vector<Mat> p;
        for (double th : thetas)
            p.push_back(panel(f2u(gaborResponse(srcF, sigma, th, lambda, gamma)),
                              "respon " + to_string((int)th) + " deg"));
        show("3_gabor_orientasi", row(p));
    }

    // -----------------------------------------------------------------
    // [4] GABOR FREKUENSI: respon citra pada beberapa lambda (sudut tetap 0)
    {
        double lambdas[4] = {6, 10, 16, 24};   // kecil = frekuensi tinggi
        vector<Mat> p;
        for (double lm : lambdas)
            p.push_back(panel(f2u(gaborResponse(srcF, sigma, 0, lm, gamma)),
                              "lambda " + to_string((int)lm)));
        show("4_gabor_frekuensi", row(p));
    }

    destroyAllWindows();
    cout << "\n=== SELESAI. Total " << stepCount << " figur ditampilkan. ===\n";
    return 0;
}
