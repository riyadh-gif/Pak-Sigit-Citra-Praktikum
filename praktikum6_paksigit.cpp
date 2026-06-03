// =====================================================================
//  PRAKTIKUM 6 - NEIGHBORHOOD OPERATOR (MORFOLOGI + DISTANCE TRANSFORM)
//  Satu program, sekali jalan, semua figur kelar.
//
//  Cara pakai:
//   1. Taruh gambar "lena.bmp" (atau Lenna.png / lena.jpg) di folder yang
//      sama. Kalau tidak ada, program tetap jalan pakai gambar pengganti.
//   2. (Opsional) taruh gambar paint 200x200 namanya "gambar.png".
//      Kalau tidak ada, dibuat otomatis (ada lubang & noise buat demo).
//   3. Compile, jalankan. Tiap window muncul -> SCREENSHOT -> pencet
//      tombol apa saja (Enter/spasi) untuk lanjut ke figur berikutnya.
//
//  Tiap panel sudah berlabel (3x3, N=5, ELLIPSE, dst.) dan judul window
//  menjelaskan figurnya, jadi langsung siap di-paste ke laporan.
//
//  Compile (Windows / MinGW contoh):
//   g++ praktikum6_all.cpp -o praktikum6 -std=c++17 \
//       -I"C:/opencv/include" -L"C:/opencv/lib" -lopencv_world4xx
//  Atau lewat Visual Studio: pastikan C++17 aktif.
// =====================================================================

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

int stepCount = 0;

// ---------------------------------------------------------------------
// Helper: ubah ke citra biner (sesuai template praktikum)
Mat toBinary(const Mat& src) {
    Mat g, bw;
    cvtColor(src, g, COLOR_BGR2GRAY);
    threshold(g, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
    return bw;
}

// Helper: operasi morfologi. N = jumlah pengulangan operasi.
// Untuk DILATE/ERODE -> N iterasi. Untuk OPEN/CLOSE -> N kali open/close penuh.
Mat morph(const Mat& bw, int op, int shape, int k, int N) {
    Mat el = getStructuringElement(shape, Size(k, k), Point(k / 2, k / 2));
    Mat dst = bw.clone();
    for (int i = 0; i < N; i++)
        morphologyEx(dst, dst, op, el);
    return dst;
}

// Helper: distance transform -> citra 8-bit yang siap ditampilkan
Mat distImg(const Mat& bw) {
    Mat d; distanceTransform(bw, d, DIST_L2, 3);
    normalize(d, d, 0, 1.0, NORM_MINMAX);
    Mat u; d.convertTo(u, CV_8U, 255);
    return u;
}

// Helper: bikin 1 panel berukuran seragam + judul kecil
Mat panel(const Mat& img, const string& label, int sz = 260) {
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

// Gabung panel
Mat row(vector<Mat> p) { Mat o; hconcat(p, o); return o; }
Mat grid2x2(vector<Mat> p) {
    Mat top, bot, o;
    hconcat(vector<Mat>{p[0], p[1]}, top);
    hconcat(vector<Mat>{p[2], p[3]}, bot);
    vconcat(vector<Mat>{top, bot}, o);
    return o;
}

// Tampilkan figur, tunggu tombol untuk lanjut (kamu screenshot di sini)
void show(const string& tag, const Mat& fig) {
    stepCount++;
    string num = (stepCount < 10 ? "0" : "") + to_string(stepCount);
    cout << "[" << num << "] " << tag << "   (screenshot, lalu tekan tombol untuk lanjut)\n";
    imshow(tag, fig);
    waitKey(0);
    destroyWindow(tag);
}

// ---------------------------------------------------------------------
// Satu operasi -> 4 figur (ukuran, tipe SE, iterasi, citra paint)
void runOp(const string& name, int op, const Mat& lena, const Mat& shapes) {
    // a) Variasi UKURAN structuring element (CROSS, N=1)
    show(name + "_ukuran", grid2x2({
        panel(morph(lena, op, MORPH_CROSS, 3,  1), "3x3"),
        panel(morph(lena, op, MORPH_CROSS, 7,  1), "7x7"),
        panel(morph(lena, op, MORPH_CROSS, 11, 1), "11x11"),
        panel(morph(lena, op, MORPH_CROSS, 15, 1), "15x15")
        }));

    // b) Variasi TIPE structuring element (ukuran 7x7, N=1)
    show(name + "_tipe", row({
        panel(morph(lena, op, MORPH_CROSS,   7, 1), "CROSS"),
        panel(morph(lena, op, MORPH_RECT,    7, 1), "RECT"),
        panel(morph(lena, op, MORPH_ELLIPSE, 7, 1), "ELLIPSE")
        }));

    // c) Variasi ITERASI N (CROSS, 7x7)
    show(name + "_iterasi", row({
        panel(morph(lena, op, MORPH_CROSS, 7, 1),  "N=1"),
        panel(morph(lena, op, MORPH_CROSS, 7, 5),  "N=5"),
        panel(morph(lena, op, MORPH_CROSS, 7, 10), "N=10")
        }));

    // d) Citra paint 200x200 (CROSS, 7x7)
    show(name + "_paint", row({
        panel(shapes,                              "asli"),
        panel(morph(shapes, op, MORPH_CROSS, 7, 1), "N=1"),
        panel(morph(shapes, op, MORPH_CROSS, 7, 5), "N=5")
        }));
}

// ---------------------------------------------------------------------
// Loader: coba beberapa nama file, balikan kosong kalau tidak ada
Mat loadFirst(vector<string> names) {
    for (auto& n : names) {
        Mat m = imread(n, IMREAD_COLOR);
        if (!m.empty()) { cout << "  -> pakai: " << n << "\n"; return m; }
    }
    return Mat();
}

// Gambar pengganti Lena (blob biner) kalau file tidak ditemukan
Mat fakeLena() {
    Mat g(360, 360, CV_8UC1);
    randu(g, 0, 255);
    GaussianBlur(g, g, Size(15, 15), 0);
    Mat c; cvtColor(g, c, COLOR_GRAY2BGR);
    return c;
}

// Gambar paint pengganti (shapes + lubang + noise) kalau tidak ditemukan
Mat fakeShapes() {
    Mat img = Mat::zeros(200, 200, CV_8UC3);
    Scalar w(255, 255, 255), b(0, 0, 0);
    circle(img, Point(45, 50), 28, w, -1);                       // lingkaran
    rectangle(img, Point(110, 25), Point(180, 75), w, -1);       // kotak
    rectangle(img, Point(140, 42), Point(155, 58), b, -1);       // lubang di kotak (buat closing)
    vector<Point> tri = { Point(45, 175), Point(15, 120), Point(78, 120) };
    fillConvexPoly(img, tri, w);                                 // segitiga
    ellipse(img, Point(150, 150), Size(38, 26), 0, 0, 360, w, -1); // elips
    circle(img, Point(100, 105), 3, w, -1);                      // noise kecil (buat opening)
    circle(img, Point(190, 105), 2, w, -1);
    return img;
}

// ---------------------------------------------------------------------
int main() {
    cout << "Memuat gambar...\n";
    Mat lenaSrc = loadFirst({ "lena.bmp", "Lenna.png", "lena.jpg", "lena.png" });
    if (lenaSrc.empty()) { cout << "  -> lena tidak ada, pakai gambar pengganti.\n"; lenaSrc = fakeLena(); }

    Mat shapesSrc = loadFirst({ "gambar.png", "shapes.png", "paint.png" });
    if (shapesSrc.empty()) { cout << "  -> gambar paint tidak ada, dibuat otomatis.\n"; shapesSrc = fakeShapes(); }

    Mat lena = toBinary(lenaSrc);
    Mat shapes = toBinary(shapesSrc);

    cout << "\n=== MULAI: tiap window -> screenshot -> tekan tombol untuk lanjut ===\n\n";

    // 1-4. Morfologi
    runOp("1_dilasi", MORPH_DILATE, lena, shapes);
    runOp("2_erosi", MORPH_ERODE, lena, shapes);
    runOp("3_opening", MORPH_OPEN, lena, shapes);
    runOp("4_closing", MORPH_CLOSE, lena, shapes);

    // 5. Distance Transform
    show("5_distance_lena", row({
        panel(lena, "binary"),
        panel(distImg(lena), "distance")
        }));
    show("5_distance_paint", row({
        panel(shapesSrc, "source"),
        panel(shapes, "binary"),
        panel(distImg(shapes), "distance")
        }));

    destroyAllWindows();
    cout << "\n=== SELESAI. Total " << stepCount << " figur ditampilkan. ===\n";
    return 0;
}
