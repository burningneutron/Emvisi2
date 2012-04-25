#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#if defined WIN32
#if defined _DEBUG
#pragma comment(lib,"opencv_core231d.lib")
#pragma comment(lib,"opencv_imgproc231d.lib")
#pragma comment(lib,"opencv_highgui231d.lib")
#pragma comment(lib, "opencv_features2d231d.lib")
#pragma comment(lib, "opencv_flann231d.lib")
#pragma comment(lib, "opencv_calib3d231d.lib")
#pragma comment(lib, "opencv_contrib231d.lib")
#pragma comment(lib, "opencv_legacy231d.lib")
#pragma comment(lib, "opencv_video231d.lib")
#else
#pragma comment(lib,"opencv_core231.lib")
#pragma comment(lib,"opencv_imgproc231.lib")
#pragma comment(lib,"opencv_highgui231.lib")
#pragma comment(lib, "opencv_features2d231.lib")
#pragma comment(lib, "opencv_flann231.lib")
#pragma comment(lib, "opencv_calib3d231.lib")
#pragma comment(lib, "opencv_contrib231.lib")
#pragma comment(lib, "opencv_legacy231.lib")
#pragma comment(lib, "opencv_video231.lib")
#endif

#pragma warning(disable: 4251)
#pragma warning(disable: 4996)
#endif

int main()
{
   // Open the video file
    cv::VideoCapture capture(0);
   // check if video successfully opened
   if (!capture.isOpened())
      return 0;
   // current video frame
   cv::Mat frame; 
   // foreground binary image
   cv::Mat foreground;
   cv::namedWindow("Extracted Foreground");
   // The Mixture of Gaussian object
   // used with all default parameters
   cv::BackgroundSubtractorMOG2 mog;
   bool stop(false);
   // for all frames in video
   while (!stop) {
      // read next frame if any
      if (!capture.read(frame))
         break;
      // update the background
      // and return the foreground
      mog(frame,foreground,0.005);
      // Complement the image 
	  cv::threshold(foreground,foreground,
                    128,255,cv::THRESH_BINARY_INV);
      // show foreground
      cv::imshow("Extracted Foreground",foreground);
      // introduce a delay
      // or press key to stop
      if (cv::waitKey(10)>=0)
            stop= true;
   }
}