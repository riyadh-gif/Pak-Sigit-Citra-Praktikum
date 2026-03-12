#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

// Mouse callback function
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        Mat* img = (Mat*)userdata;

        // Ambil nilai pixel
        Vec3b color = img->at<Vec3b>(y, x);

        int blue = color[0];
        int green = color[1];
        int red = color[2];

        cout << "Klik pada posisi (" << x << ", " << y << ")" << endl;
        cout << "Nilai RGB = (" << red << ", " << green << ", " << blue << ")" << endl;
    }
}

int main(int argc, char** argv)
{
    // Read image from file
    Mat img = imread("wowo.jpg");

    // Jika gambar gagal dibuka
    if (img.empty())
    {
        cout << "Error loading the image" << endl;
        return -1;
    }

    // Create window
    namedWindow("My Window", 1);

    // Set mouse callback dan kirim gambar sebagai userdata
    setMouseCallback("My Window", CallBackFunc, &img);

    // Show image
    imshow("My Window", img);

    // Wait key
    waitKey(0);

    return 0;
}
