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
 */
#ifndef EMVISI2_H
#define EMVISI2_H

#include "fwncc.h"
#include "imstat.h"
#include "NccHisto.h"

class EMVisi2 {
public:
	bool save_images;

	EMVisi2();

	bool init();
	int setModel(const IplImage *im1, const IplImage *mask=0);
	int setTarget(const IplImage *target);
	void iterate();
	void smooth(float amount, float threshold);
	void run(int nbIter=3, float smooth_amount=2.5, float smooth_threshold=.0001);
	float process_pixel(const float *rgb, const float *frgb, const float dl, const float nccv, const float ncch, float *proba, float *visi_proba);
	void reset_gaussians();

	cv::Mat proba, visi_proba;

	bool recycle;
	float PF;
	static const int ncc_size = 25;

	cv::Mat prod_f, prod_g;

protected:
	FNcc fncc;
	NccHisto ncc_h;
	NccHisto ncc_v;
	cv::Mat im1f;
	cv::Mat mask;

	cv::Mat visible;
	cv::Mat hidden;
	cv::Mat ncc, sum;
	cv::Mat ratio;
	cv::Mat nccproba_v;
	cv::Mat nccproba_h;
	cv::Mat dx,dy,diffusion;
	int iteration;

	cv::Mat _im2;

	cv::Mat dL;

#define NB_VISI_GAUSSIANS 2
#define NB_OCCL_GAUSSIANS 2 
#define NB_GAUSSIANS (NB_VISI_GAUSSIANS+NB_OCCL_GAUSSIANS)

	MultiGaussian3<float> visi_g[NB_VISI_GAUSSIANS];
	MultiGaussian3<float> occl_g[NB_OCCL_GAUSSIANS];
	float weights[NB_GAUSSIANS+1];

	float uniform_resp;
};

void log_save(const char *fn, IplImage *im);
void a_save(const char *fn, IplImage *im);
void save_proba(const char *fn, IplImage *im);
void scale_save(const char *fn, IplImage *im, double scale=-1, double shift=-1);

#endif
