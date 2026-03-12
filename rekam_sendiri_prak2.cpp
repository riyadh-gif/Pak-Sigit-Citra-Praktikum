#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    VideoCapture cap(0);

    if (!cap.isOpened())
    {
        cout << "Camera tidak bisa dibuka" << endl;
        return -1;
    }

    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);

    VideoWriter video("record.avi",
        VideoWriter::fourcc('M', 'J', 'P', 'G'),
        20,
        Size(width, height));

    if (!video.isOpened())
    {
        cout << "VideoWriter gagal dibuat!" << endl;
        return -1;
    }

    Mat frame;

    cout << "Recording 10 detik..." << endl;

    double start = (double)getTickCount();

    while (true)
    {
        cap >> frame;

        if (frame.empty())
            break;

        imshow("Camera", frame);

        video.write(frame);   // MENULIS FRAME KE VIDEO

        double time = ((double)getTickCount() - start) / getTickFrequency();

        if (time > 10)   // rekam 10 detik
            break;

        if (waitKey(30) == 27)
            break;
    }

    cap.release();
    video.release();
    destroyAllWindows();

    cout << "Video selesai disimpan" << endl;

    return 0;
}
