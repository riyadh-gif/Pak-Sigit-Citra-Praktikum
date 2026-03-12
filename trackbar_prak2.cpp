#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

/// Global variables
int threshold_value = 0;
int threshold_type = 3;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

Mat src, src_gray, dst;
const char* window_name = "Threshold Demo";

const char* trackbar_type =
"Type:\n"
"0: Binary\n"
"1: Binary Inverted\n"
"2: Truncate\n"
"3: To Zero\n"
"4: To Zero Inverted";

const char* trackbar_value = "Value";

/// Function prototype
void Threshold_Demo(int, void*);

/**
 * main function
 */
int main(int argc, char** argv)
{
    String imageName("stuff.jpg");

    if (argc > 1)
    {
        imageName = argv[1];
    }

    // Load image
    src = imread(imageName, IMREAD_COLOR);

    if (src.empty())
    {
        return -1;
    }

    // Convert to grayscale
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    // Create window
    namedWindow(window_name, WINDOW_AUTOSIZE);

    // Create trackbars
    createTrackbar(trackbar_type, window_name, &threshold_type, max_type, Threshold_Demo);
    createTrackbar(trackbar_value, window_name, &threshold_value, max_value, Threshold_Demo);

    // Initial call
    Threshold_Demo(0, 0);

    // Wait until ESC pressed
    for (;;)
    {
        char c = (char)waitKey(20);
        if (c == 27)
        {
            break;
        }
    }

    return 0;
}

/**
 * Threshold function
 */
void Threshold_Demo(int, void*)
{
    threshold(src_gray, dst, threshold_value, max_BINARY_value, threshold_type);
    imshow(window_name, dst);
}
