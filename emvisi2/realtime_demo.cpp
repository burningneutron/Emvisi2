#include <iostream>
#include "emvisi2.h"
#include "growmat.h"
#include "ncc_proba.h"

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
#else
#pragma comment(lib,"opencv_core231.lib")
#pragma comment(lib,"opencv_imgproc231.lib")
#pragma comment(lib,"opencv_highgui231.lib")
#pragma comment(lib, "opencv_features2d231.lib")
#pragma comment(lib, "opencv_flann231.lib")
#pragma comment(lib, "opencv_calib3d231.lib")
#endif

#pragma warning(disable: 4251)
#pragma warning(disable: 4996)
#endif

Mat normalizeMat(const Mat input)
{
	double minVal, maxVal;
	minMaxLoc(input, &minVal, &maxVal);

	Mat normMat = (input - minVal) / (maxVal - minVal) * 255;

	Mat output;

	normMat.convertTo(output, CV_8U);

	return output;
}

int main(int argc, char *argv[])
{
	double st, et;
	EMVisi2 emv;

	cout << "Open camera ..." << endl;
	VideoCapture capture;
	capture.open(0);
	assert(capture.isOpened());

	Mat frame;
	capture >> frame;

	IplImage frameStub = frame;

	cout << "emvisi2 init ..." << endl;
	if (!emv.init()) {
		cout << "EMVisi2::init() failed.\n";
		return -1;
	}

	cout << "emvisi2 set model ..." << endl;
	st = getTickCount();

	emv.setModel(&frameStub, 0);

	et = getTickCount();
	cout << " init background takes: " << (et-st)/getTickFrequency() << endl;

	while(1){
		capture >> frame;
		frameStub = frame;

		st = getTickCount();

		emv.setTarget(&frameStub);
		const int niter=10;
		emv.run(niter, 0);

		et = getTickCount();
		cout << " fg segmentation takes: " << (et-st)/getTickFrequency() << endl;

		imshow("video", frame);

		Mat fgImage = normalizeMat(emv.proba);
		imshow("seg", fgImage);

		waitKey(1);
	}

	return 0;
}