// ============================================================================
// Praktikum 10 - Ekstraksi Fitur Warna : MOMENT WARNA (Tugas 1 + Tugas 2)
// Satu kode untuk kedua tugas (gaya seperti program Histogram).
//
//   TUGAS 1 (jalankan & amati): ditampilkan dari gambar ACUAN (gambar pertama).
//     - Gambar dibagi 3x3 = 9 blok, tiap blok 9 fitur (mean/std/skew HSV).
//     - Grid + label B1..B9 ditampilkan, dan tabel 81 fitur dicetak ke konsol.
//
//   TUGAS 2 (studi kasus 10 gambar):
//     - Semua gambar di folder diproses, fitur disimpan ke .txt per gambar.
//     - fitur_tomat.csv  : tabel fitur 10 gambar (81 kolom).
//     - selisih_tomat.csv: selisih mutlak tiap gambar terhadap acuan (Tomat_1).
//
// Build (VS/Windows): Build & Run.
// Build (Linux): g++ praktikum10_moment.cpp -o cm `pkg-config --cflags --libs opencv4`
// ============================================================================

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdio>
using namespace cv;
using namespace std;

// Pemisah kolom CSV. Jika Excel (locale Indonesia) tidak memisah kolom, ganti ';'
static const char SEP = ',';

void channelMoments(const Mat& ch, double& mean, double& stdev, double& skew)
{
    Scalar m, s;
    meanStdDev(ch, m, s);
    mean = m[0]; stdev = s[0];
    double sum3 = 0.0; int N = ch.rows * ch.cols;
    for (int r = 0; r < ch.rows; r++)
        for (int c = 0; c < ch.cols; c++) {
            double d = (double)ch.at<uchar>(r, c) - mean;
            sum3 += d * d * d;
        }
    double denom = (double)N * std::pow(stdev, 3.0);
    skew = (std::fabs(denom) > 1e-9) ? (sum3 / denom) : 0.0;   // hindari NaN
}

// 81 fitur (9 blok x 9), urutan per blok: mH,sH,kH, mS,sS,kS, mV,sV,kV
vector<double> computeFeatures(const Mat& src)
{
    Mat hsv; cvtColor(src, hsv, COLOR_BGR2HSV);
    const int grid = 3;
    int bh = hsv.rows / grid, bw = hsv.cols / grid;
    vector<double> f;
    for (int row = 0; row < grid; row++)
        for (int col = 0; col < grid; col++) {
            Rect rect(col * bw, row * bh, bw, bh);
            vector<Mat> ch; split(hsv(rect), ch);
            double mH, sH, kH, mS, sS, kS, mV, sV, kV;
            channelMoments(ch[0], mH, sH, kH);
            channelMoments(ch[1], mS, sS, kS);
            channelMoments(ch[2], mV, sV, kV);
            mH /= 180.0; sH /= 180.0; mS /= 255.0; sS /= 255.0; mV /= 255.0; sV /= 255.0;
            f.insert(f.end(), { mH,sH,kH, mS,sS,kS, mV,sV,kV });
        }
    return f;
}

vector<string> featureHeader()
{
    vector<string> feat = { "Mean_H","Std_H","Skew_H","Mean_S","Std_S","Skew_S","Mean_V","Std_V","Skew_V" };
    vector<string> h;
    for (int b = 1; b <= 9; b++) for (auto& f : feat) h.push_back("B" + to_string(b) + "_" + f);
    return h;
}

string baseName(const string& path)
{
    size_t slash = path.find_last_of("/\\");
    string fn = (slash == string::npos) ? path : path.substr(slash + 1);
    size_t dot = fn.find_last_of('.');
    return (dot == string::npos) ? fn : fn.substr(0, dot);
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

// TUGAS 1: gambar grid 3x3 + label, dan cetak tabel 9 blok x 9 fitur ke konsol.
Mat tampilkanTugas1(const Mat& src, const vector<double>& f, const string& label)
{
    Mat shown = src.clone();
    int grid = 3, bh = src.rows / grid, bw = src.cols / grid, n = 1;
    for (int row = 0; row < grid; row++)
        for (int col = 0; col < grid; col++) {
            Rect rect(col * bw, row * bh, bw, bh);
            rectangle(shown, rect, Scalar(0, 255, 0), 1);
            putText(shown, "B" + to_string(n), Point(rect.x + 4, rect.y + 18),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1, LINE_AA);
            n++;
        }

    cout << "\n=== TUGAS 1: fitur momen gambar acuan (" << label << ") ===\n";
    cout << "Blok | Mean_H Std_H  Skew_H | Mean_S Std_S  Skew_S | Mean_V Std_V  Skew_V\n";
    for (int b = 0; b < 9; b++) {
        int i = b * 9;
        printf("B%-3d | %.4f %.4f %7.3f | %.4f %.4f %7.3f | %.4f %.4f %7.3f\n",
            b + 1, f[i], f[i + 1], f[i + 2], f[i + 3], f[i + 4], f[i + 5], f[i + 6], f[i + 7], f[i + 8]);
    }
    return shown;
}

int main()
{
    string folder = "C:/Users/riyadh/Downloads/tomat/";

    vector<String> files;
    glob(folder + "*", files, false);
    sort(files.begin(), files.end(),
        [](const String& a, const String& b) { return naturalLess(string(a), string(b)); });

    vector<string>         names;
    vector<vector<double>> feats;
    Mat refImg; string refName;

    for (const auto& fpath : files) {
        Mat img = imread(string(fpath), IMREAD_COLOR);
        if (img.empty()) continue;
        string base = baseName(string(fpath));
        vector<double> f = computeFeatures(img);
        names.push_back(base);
        feats.push_back(f);
        if (refImg.empty()) { refImg = img.clone(); refName = base; }  // gambar acuan

        int idx = (int)feats.size();
        ofstream t(folder + "moment_Tomat_" + to_string(idx) + ".txt");
        vector<string> H = featureHeader();
        t << "File asli: " << base << "\n";
        for (size_t k = 0; k < f.size(); k++) t << H[k] << " = " << f[k] << "\n";
        t.close();
    }
    if (feats.empty()) { cerr << "ERROR: tidak ada gambar di " << folder << endl; return -1; }

    vector<string> header = featureHeader();

    // ---- TUGAS 2: CSV fitur ----
    {
        ofstream csv(folder + "fitur_tomat.csv");
        csv << "Nama_Gambar" << SEP << "File_Asli";
        for (auto& h : header) csv << SEP << h;
        csv << "\n";
        for (size_t r = 0; r < feats.size(); r++) {
            csv << "Tomat_" << (r + 1) << SEP << names[r];
            for (double v : feats[r]) csv << SEP << v;
            csv << "\n";
        }
    }
    // ---- TUGAS 2: CSV selisih terhadap acuan (Tomat_1) ----
    {
        ofstream csv(folder + "selisih_tomat.csv");
        csv << "Nama_Gambar" << SEP << "File_Asli";
        for (auto& h : header) csv << SEP << h;
        csv << "\n";
        for (size_t r = 0; r < feats.size(); r++) {
            csv << "Tomat_" << (r + 1) << SEP << names[r];
            for (size_t k = 0; k < feats[r].size(); k++)
                csv << SEP << std::fabs(feats[r][k] - feats[0][k]);
            csv << "\n";
        }
    }

    cout << "Gambar diproses: " << feats.size() << ". Acuan = Tomat_1 (" << names[0] << ")\n";
    cout << "Tersimpan di " << folder << " : moment_Tomat_1..N.txt, fitur_tomat.csv, selisih_tomat.csv\n";

    // ---- TUGAS 1: tampilkan & cetak tabel untuk gambar acuan ----
    Mat grid = tampilkanTugas1(refImg, feats[0], "Tomat_1 / " + refName);
    imwrite(folder + "moment_grid.png", grid);
    namedWindow("Tugas 1 - Grid 3x3 (gambar acuan)", WINDOW_AUTOSIZE);
    imshow("Tugas 1 - Grid 3x3 (gambar acuan)", grid);
    cout << "\nTekan sembarang tombol untuk keluar.\n";
    waitKey(0);
    return 0;
}
