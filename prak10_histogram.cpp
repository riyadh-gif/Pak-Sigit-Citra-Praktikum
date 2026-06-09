// ============================================================================
// Praktikum 10 - Ekstraksi Fitur Warna : HISTOGRAM
// Menyelesaikan Tugas 1 dan Tugas 2 dalam satu program.
//
//   Tugas 1: Histogram 256 bin (8-bit) untuk kanal B, G, R.
//   Tugas 2: Histogram 16 bin  (4-bit) untuk kanal B, G, R.
//            (256 tingkat intensitas dikelompokkan jadi 16, jadi tiap bin
//             mewakili 16 tingkat = kuantisasi warna 4-bit.)
//
// Perbedaan utama hanya pada jumlah bin (histSize). Program menampilkan gambar
// asli, histogram 256 bin, dan histogram 16 bin, lalu menyimpan keduanya.
//
// Build (VS/Windows): Build & Run (path sudah di-set).
// Build (Linux): g++ praktikum10_histogram.cpp -o hist `pkg-config --cflags --libs opencv4`
// ============================================================================

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
using namespace cv;
using namespace std;

// Hitung dan gambar histogram B, G, R untuk jumlah bin tertentu.
Mat buatHistogram(const vector<Mat>& bgr, int histSize, const string& judul)
{
    // Rentang nilai intensitas 0..255
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true, accumulate = false;

    Mat b_hist, g_hist, r_hist;
    calcHist(&bgr[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    // Kanvas histogram
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    // Normalisasi tinggi grafik agar muat di kanvas
    normalize(b_hist, b_hist, 0, hist_h, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, hist_h, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, hist_h, NORM_MINMAX, -1, Mat());

    // Gambar garis tiap kanal
    for (int i = 1; i < histSize; i++)
    {
        line(histImage,
            Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
            Scalar(255, 0, 0), 2, 8, 0);   // Biru
        line(histImage,
            Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
            Scalar(0, 255, 0), 2, 8, 0);   // Hijau
        line(histImage,
            Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
            Scalar(0, 0, 255), 2, 8, 0);   // Merah
    }

    // Judul di pojok kiri atas
    putText(histImage, judul, Point(10, 22), FONT_HERSHEY_SIMPLEX, 0.55,
        Scalar(255, 255, 255), 1, LINE_AA);
    return histImage;
}

int main()
{
    string outDir = "C:/Users/riyadh/Downloads/";
    string path = outDir + "pejuangkemerdekaan.jpg";

    Mat src = imread(path, IMREAD_COLOR);
    if (src.empty()) { cerr << "ERROR: gambar tidak ditemukan: " << path << endl; return -1; }

    // Pisahkan menjadi 3 kanal B, G, R
    vector<Mat> bgr;
    split(src, bgr);

    // Tugas 1 dan Tugas 2
    Mat hist256 = buatHistogram(bgr, 256, "Histogram 8-bit (256 bin)");
    Mat hist16 = buatHistogram(bgr, 16, "Histogram 4-bit (16 bin)");

    // Tampilkan
    namedWindow("Source Image", WINDOW_AUTOSIZE);
    imshow("Source Image", src);
    imshow("Histogram 8-bit (256 bin)", hist256);
    imshow("Histogram 4-bit (16 bin)", hist16);

    // Simpan untuk laporan
    imwrite(outDir + "hist_256bin.png", hist256);
    imwrite(outDir + "hist_16bin.png", hist16);
    cout << "Tersimpan: hist_256bin.png dan hist_16bin.png di " << outDir << "\n";
    cout << "Tekan sembarang tombol untuk keluar.\n";

    waitKey(0);
    return 0;
}
