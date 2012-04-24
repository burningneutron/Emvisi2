/*
emvisi2 makes background subtraction robust to illumination changes.
Copyright (C) 2008 Julien Pilet, Christoph Strecha, and Pascal Fua.

This file is part of emvisi2.

emvisi2 is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

emvisi2 is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with emvisi2.  If not, see <http://www.gnu.org/licenses/>.


For more information about this code, see our paper "Making Background
Subtraction Robust to Sudden Illumination Changes".
*/
/*
* Julien Pilet, Feb 2008
* Packaged on Nov 2008
*/
#include <iostream>
#include "emvisi2.h"
#include "growmat.h"
#include "ncc_proba.h"

#include <opencv2/opencv.hpp>

// To enable graphcut support, 
//  - download Yuri Boykov's implementation.
//    http://www.adastral.ucl.ac.uk/~vladkolm/software/maxflow-v3.0.src.tar.gz
//  - edit the Makefile

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


void usage(char *str)
{
	cerr << "usage: " << str << " [-v] <background> <input frame> [<mask>]\n";
	cerr << "	use -v for more verbosity and more intermediate images saved.\n";
	exit(-1);
}


int main(int argc, char *argv[])
{
	EMVisi2 emv;
	int nim=0;
	IplImage *im[3] = {0,0,0};

	// parse command line
	for (int narg=1; narg<argc; ++narg) {
		if (strcmp(argv[narg],"-v")==0) {
			emv.save_images=true;
		} else {
			im[nim] = cvLoadImage(argv[narg], (nim==2 ? 0 : -1));
			if (!im[nim]) {
				cerr << argv[narg] << ": can't load image.\n";
				exit(-2);
			}
			nim++;
		}
	}

	IplImage *im1 = im[0];
	IplImage *im2 = im[1];
	IplImage *mask = im[2];

	if (!im1 || !im2) usage(argv[0]);

	if ((im1->nChannels != im2->nChannels) || (im1->width != im2->width) ) {
		cerr << "image format or size do not match.\n";
		exit(-4);
	}
	int h = (im1->height < im2->height ? im1->height : im2->height);
	im1->height=h;
	im2->height=h;

	cout << "Initialization.. ";

	// EMVisi2 setup
	if (!emv.init()) {
		cerr << "EMVisi2::init() failed.\n";
		return -1;
	}
	emv.setModel(im1, mask);


	cout << "setTarget... ";
	emv.setTarget(im2);

	const int niter=32;
	emv.run(niter, 0);

#ifdef WITH_GRAPHCUT
	emv.smooth(2.4, 0.001);
	float gc_duration = iterations.duration() - it_duration;
#endif

#ifdef WITH_GRAPHCUT
	cout << "graph cut computed in " << gc_duration << " ms.\n";
#endif


	IplImage probaStub = emv.proba;

	save_proba("final_proba.png", &probaStub);
	cvReleaseImage(&im1);
	cvReleaseImage(&im2);
	cvReleaseImage(&mask);
	return 0;
}

