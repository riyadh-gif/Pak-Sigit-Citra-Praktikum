#include <Windows.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;
using namespace std;

int main(int argc, const char** argv)
{
    // Load the image from file
    Mat LoadedImage;

    // Load image Lenna.png from project directory
    LoadedImage = imread("Lenna.png", IMREAD_COLOR);

    // Check if image loaded successfully
    if (LoadedImage.empty())
    {
        printf("Error: Image not found!\n");
        return -1;
    }

    // Show loaded image
    namedWindow("Step 1 image loaded", WINDOW_AUTOSIZE);
    imshow("Step 1 image loaded", LoadedImage);
    waitKey(1000);

    // Save the loaded image
    imwrite("Step1.JPG", LoadedImage);

    // Create rectangle (x, y, width, height)
    Rect Rec(50, 50, 300, 250);

    // Draw rectangle on image
    rectangle(LoadedImage, Rec, Scalar(255, 0, 0), 2, 8, 0);

    // Show image with rectangle
    namedWindow("Step 2 draw Rectangle", WINDOW_AUTOSIZE);
    imshow("Step 2 draw Rectangle", LoadedImage);
    waitKey(1000);

    // Save step 2
    imwrite("Step2.JPG", LoadedImage);

    // Select ROI (Region of Interest)
    Mat Roi = LoadedImage(Rec);

    namedWindow("Step 3 Draw selected Roi", WINDOW_AUTOSIZE);
    imshow("Step 3 Draw selected Roi", Roi);
    waitKey(1000);

    // Save ROI
    imwrite("Step3.JPG", Roi);

    // Rectangle target position to paste ROI
    Rect WhereRec(0, 0, Roi.cols, Roi.rows);

    // Copy ROI to top-left corner
    Roi.copyTo(LoadedImage(WhereRec));

    namedWindow("Step 4 Final result", WINDOW_AUTOSIZE);
    imshow("Step 4 Final result", LoadedImage);
    waitKey(1000);

    // Save final result
    imwrite("Step4.JPG", LoadedImage);

    return 0;
}
