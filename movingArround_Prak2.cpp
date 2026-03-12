#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int g_slider_position = 0;
VideoCapture g_capture;

// Callback function untuk trackbar
void onTrackbarSlide(int pos, void*)
{
    g_capture.set(CAP_PROP_POS_FRAMES, pos);
}

int main(int argc, char** argv)
{
    namedWindow("AVI", WINDOW_AUTOSIZE);

    // Membuka video
    g_capture.open("video.avi");

    if (!g_capture.isOpened())
    {
        cout << "Error: Video tidak bisa dibuka!" << endl;
        return -1;
    }

    int frames = (int)g_capture.get(CAP_PROP_FRAME_COUNT);

    // Membuat trackbar jika jumlah frame diketahui
    if (frames > 0)
    {
        createTrackbar(
            "Position",
            "AVI",
            &g_slider_position,
            frames,
            onTrackbarSlide
        );
    }

    Mat frame;

    while (true)
    {
        g_capture >> frame;

        if (frame.empty())
            break;

        imshow("AVI", frame);

        char c = (char)waitKey(25);
        if (c == 27) // ESC
            break;
    }

    g_capture.release();
    destroyWindow("AVI");

    return 0;
}
