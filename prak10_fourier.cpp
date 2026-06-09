// ============================================================================
// Praktikum 10 - Ekstraksi Fitur Warna : FOURIER (Tugas 1 + Tugas 2)
// Satu kode untuk kedua tugas. Memakai DFT OpenCV (cepat) sebagai pengganti
// DFT manual di modul yang sangat lambat.
//
//   TUGAS 1 (jalankan & amati): tampilkan SPEKTRUM FOURIER 2-D gambar acuan.
//     - Citra diubah ke domain frekuensi, magnitudo di-log + digeser ke pusat.
//     - Pusat terang = frekuensi rendah (energi dominan).
//
//   TUGAS 2 (fitur 1-D + pencocokan):
//     - Spektrum diringkas jadi grid 8x8 lalu diratakan -> vektor 1-D 64 fitur
//       (F1..F64), dinormalisasi L2.
//     - fourier_fitur.csv : tabel 64 fitur untuk semua gambar.
//     - fourier_match.csv : skor SQUARE DIFFERENCE tiap gambar terhadap acuan
//       (Tomat_1). Makin kecil skor, makin mirip.
//
// Build (VS/Windows): Build & Run.
// Build (Linux): g++ praktikum10_fourier.cpp -o ft `pkg-config --cflags --libs opencv4`
// ============================================================================

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cctype>
using namespace cv;
using namespace std;

static const char SEP = ',';

// Geser kuadran spektrum agar DC (frekuensi nol) berada di tengah.
void shiftDFT(Mat& mag)
{
    int cx = mag.cols / 2, cy = mag.rows / 2;
    Mat q0(mag, Rect(0, 0, cx, cy)), q1(mag, Rect(cx, 0, cx, cy));
    Mat q2(mag, Rect(0, cy, cx, cy)), q3(mag, Rect(cx, cy, cx, cy));
    Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
}

// Magnitudo log spektrum (untuk perhitungan & tampilan), sudah digeser ke pusat.
Mat logMagnitude(const Mat& gray)
{
    Mat f; gray.convertTo(f, CV_32F);
    Mat planes[] = { f, Mat::zeros(f.size(), CV_32F) };
    Mat complexI; merge(planes, 2, complexI);
    dft(complexI, complexI);
    split(complexI, planes);
    Mat mag; magnitude(planes[0], planes[1], mag);
    mag += Scalar::all(1);
    log(mag, mag);
    mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2)); // pastikan genap
    shiftDFT(mag);
    return mag;
}

// TUGAS 1: spektrum 2-D untuk ditampilkan (256x256, dinormalisasi 0..1).
Mat spektrum2D(const Mat& src)
{
    Mat gray; cvtColor(src, gray, COLOR_BGR2GRAY);
    resize(gray, gray, Size(256, 256));
    Mat mag = logMagnitude(gray);
    normalize(mag, mag, 0, 1, NORM_MINMAX);
    return mag;
}

// TUGAS 2: fitur 1-D = ringkasan spektrum jadi 8x8 (64 nilai), L2-normalized.
vector<double> fourierFeature(const Mat& src)
{
    Mat gray; cvtColor(src, gray, COLOR_BGR2GRAY);
    resize(gray, gray, Size(64, 64));          // ukuran tetap -> panjang fitur konsisten
    Mat mag = logMagnitude(gray);

    Mat small; resize(mag, small, Size(8, 8));  // ringkas jadi 8x8 = 64 fitur
    vector<double> v; v.reserve(64);
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            v.push_back(small.at<float>(r, c));

    // Normalisasi L2 (vektor satuan) -> invarian terhadap energi total
    double ss = 0.0; for (double x : v) ss += x * x;
    double norm = std::sqrt(ss);
    if (norm > 1e-9) for (double& x : v) x /= norm;
    return v;                                   // 64 fitur
}

string baseName(const string& path)
{
    size_t slash = path.find_last_of("/\\");
    string fn = (slash == string::npos) ? path : path.substr(slash + 1);
    size_t dot = fn.find_last_of('.');
    return (dot == string::npos) ? fn : fn.substr(0, dot);
}

// Terima hanya file gambar masukan, lewati file output kita (moment_/fourier_/hist_/grid/spectrum).
bool isInputImage(const string& path)
{
    string b = baseName(path);
    string lb = b; transform(lb.begin(), lb.end(), lb.begin(), ::tolower);
    if (lb.rfind("moment_", 0) == 0 || lb.rfind("fourier_", 0) == 0 || lb.rfind("hist_", 0) == 0) return false;
    if (lb.find("grid") != string::npos || lb.find("spectrum") != string::npos) return false;
    string ext; size_t dot = path.find_last_of('.');
    if (dot != string::npos) ext = path.substr(dot + 1);
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" || ext == "webp");
}

bool naturalLess(const string& a, const string& b)
{
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (isdigit((unsigned char)a[i]) && isdigit((unsigned char)b[j])) {
            size_t i2 = i, j2 = j;
            while (i2 < a.size() && isdigit((unsigned char)a[i2])) i2++;
            while (j2 < b.size() && isdigit((unsigned char)b[j2])) j2++;
            long va = stol(a.substr(i, i2 - i)), vb = stol(b.substr(j, j2 - j));
            if (va != vb) return va < vb;
            i = i2; j = j2;
        }
        else { if (a[i] != b[j]) return a[i] < b[j]; i++; j++; }
    }
    return a.size() < b.size();
}

int main()
{
    string folder = "C:/Users/riyadh/Downloads/tomat/";

    vector<String> all;
    glob(folder + "*", all, false);
    vector<string> files;
    for (auto& p : all) if (isInputImage(string(p))) files.push_back(string(p));
    sort(files.begin(), files.end(),
        [](const string& a, const string& b) { return naturalLess(a, b); });

    vector<string>         names;
    vector<vector<double>> feats;
    Mat refImg; string refName;

    for (const auto& fpath : files) {
        Mat img = imread(fpath, IMREAD_COLOR);
        if (img.empty()) continue;
        names.push_back(baseName(fpath));
        feats.push_back(fourierFeature(img));
        if (refImg.empty()) { refImg = img.clone(); refName = baseName(fpath); }
    }
    if (feats.empty()) { cerr << "ERROR: tidak ada gambar di " << folder << endl; return -1; }

    // Header F1..F64
    vector<string> header;
    for (int k = 1; k <= 64; k++) header.push_back("F" + to_string(k));

    // ---- TUGAS 2a: CSV fitur Fourier 1-D ----
    {
        ofstream csv(folder + "fourier_fitur.csv");
        csv << "Nama_Gambar" << SEP << "File_Asli";
        for (auto& h : header) csv << SEP << h;
        csv << "\n";
        for (size_t r = 0; r < feats.size(); r++) {
            csv << "Tomat_" << (r + 1) << SEP << names[r];
            for (double v : feats[r]) csv << SEP << v;
            csv << "\n";
        }
    }
    // ---- TUGAS 2b: pencocokan SQUARE DIFFERENCE terhadap acuan (Tomat_1) ----
    {
        ofstream csv(folder + "fourier_match.csv");
        csv << "Nama_Gambar" << SEP << "File_Asli" << SEP << "SquareDifference_vs_Tomat_1\n";
        for (size_t r = 0; r < feats.size(); r++) {
            double sd = 0.0;
            for (size_t k = 0; k < feats[r].size(); k++) {
                double d = feats[r][k] - feats[0][k];
                sd += d * d;                       // square difference
            }
            csv << "Tomat_" << (r + 1) << SEP << names[r] << SEP << sd << "\n";
        }
    }

    cout << "Gambar diproses: " << feats.size() << ". Acuan = Tomat_1 (" << names[0] << ")\n";
    cout << "Skor SquareDifference terhadap Tomat_1 (makin kecil makin mirip):\n";
    for (size_t r = 0; r < feats.size(); r++) {
        double sd = 0.0;
        for (size_t k = 0; k < feats[r].size(); k++) { double d = feats[r][k] - feats[0][k]; sd += d * d; }
        printf("  Tomat_%-2zu (%-16s) : %.6f\n", r + 1, names[r].c_str(), sd);
    }
    cout << "Tersimpan: fourier_fitur.csv & fourier_match.csv di " << folder << "\n";

    // ---- TUGAS 1: tampilkan spektrum Fourier 2-D gambar acuan ----
    Mat spec = spektrum2D(refImg);
    Mat spec8u; spec.convertTo(spec8u, CV_8U, 255.0);
    imwrite(folder + "fourier_spectrum.png", spec8u);

    namedWindow("Gambar Acuan", WINDOW_AUTOSIZE);
    namedWindow("Spektrum Fourier 2-D (acuan)", WINDOW_AUTOSIZE);
    imshow("Gambar Acuan", refImg);
    imshow("Spektrum Fourier 2-D (acuan)", spec);
    cout << "\nTekan sembarang tombol untuk keluar.\n";
    waitKey(0);
    return 0;
}
