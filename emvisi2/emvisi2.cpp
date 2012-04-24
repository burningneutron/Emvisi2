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

void log_save(const char *fn, IplImage *im) {
	cout << "(log) ";
	cv::Mat tmp(im, true);
	cv::log(tmp, tmp);

	IplImage tmpStub = tmp;
	scale_save(fn, &tmpStub);
}
void a_save(const char *fn, IplImage *im) {
	cout << "(-log(1-x)) ";
	cv::Mat tmp(im, true);
	tmp = 1.f - tmp;
	cv::log(tmp, tmp);
	tmp = 1.f * tmp;
	IplImage tmpStub = tmp;
	scale_save(fn, &tmpStub);
}

void save_proba(const char *fn, IplImage *im) {
	char str[1024];
	_snprintf(str,1024,"log_%s",fn);
	scale_save(fn, im);
	log_save(str,im);
	_snprintf(str,1024,"exp_%s",fn);
	a_save(str,im);
}

EMVisi2::EMVisi2() {
	save_images=false;
	exp_table(0);
	recycle = false;
	//PF=.93;
	PF=.5;
	reset_gaussians();
}


void scale_save(const char *fn, IplImage *im, double scale, double shift)
{
	double sc=scale,sh=shift;
	IplImage *cvt = cvCreateImage(cvGetSize(im), IPL_DEPTH_8U, im->nChannels);
	double min= (0 - shift)/scale, max=(255-shift)/scale;
	if ((scale == -1) && (shift == -1)) {
		cvSetImageCOI(im,1);
		cvMinMaxLoc(im, &min, &max);
		cvSetImageCOI(im,0);
		sc= 255.0/(max-min);
		sh= -min*sc;
	}
	cvCvtScale(im,cvt, sc, sh);
	cout << fn << " scale: " << max << ", " << min << endl;
	cvSaveImage(fn, cvt);
	cvReleaseImage(&cvt);
}

void EMVisi2::run(int nbIter, float smooth_amount, float smooth_threshold)
{
	if (!recycle) {
		reset_gaussians();
	}
	for (int i=0;i<nbIter;i++) {
		iterate();
	}
	if (smooth_amount>0)
		smooth(smooth_amount, smooth_threshold);
}

void EMVisi2::reset_gaussians() 
{
	for (int i=0; i<NB_VISI_GAUSSIANS;i++) {
		const float max = 90;
		const float min = 0;
		visi_g[i].init_regular( (i+1)*(max-min)/(NB_VISI_GAUSSIANS+1) + min, 30*((max-min)/NB_VISI_GAUSSIANS));
	}
	for (int i=0; i<NB_OCCL_GAUSSIANS;i++) {
		const float max = 255;
		const float min = 0;
		occl_g[i].init_regular( (i+1)*(max-min)/(NB_OCCL_GAUSSIANS+1) + min, 30*((max-min)/NB_OCCL_GAUSSIANS));
	}
	for (int i=0; i<NB_GAUSSIANS+1; i++)
		weights[i] = 1.0f/(NB_GAUSSIANS+1);
}

void EMVisi2::iterate()
{
	char str[256];
	uniform_resp=0;

	for (int i=0; i<NB_VISI_GAUSSIANS; i++) {
		visi_g[i].sigma_computed=false;
	}

	for (int i=0; i<NB_OCCL_GAUSSIANS; i++) {
		occl_g[i].sigma_computed=false;
	}


	float likelihood = 0;
	for (int y=0; y<proba.rows; y++) {
		float *po = (float *)proba.ptr(y);
		float *vpo = (!visi_proba.empty() ? (float *)visi_proba.ptr(y) : 0);
		const float *nccv = (const float *)nccproba_v.ptr(y);
		const float *ncch = (const float *)nccproba_h.ptr(y);
		//const float *input = &CV_IMAGE_ELEM(im2, const float, y, 0);
		const float *input = (const float *)_im2.ptr(y);
		const float *r = (const float *)ratio.ptr(y);
		const float *dl = (const float *)dL.ptr(y);
		const unsigned char *m=0;
		if (!mask.empty()) m = (const unsigned char *) mask.ptr(y);
		for (int x=0; x<proba.cols; x++) {
			if ((!m) || m[x]) {
				float l = process_pixel(input+3*x, r+3*x, dl[x], nccv[x], ncch[x], 
					po+x, (vpo?vpo+x:0));
				if (save_images)
					likelihood += log(l);
				assert(_finite(po[x]));
			} else
				po[x]=0;
		}
	}

	if (save_images)
		cout << "L=" << likelihood << endl;

	float totN = 0;
	for (int i=0; i<NB_VISI_GAUSSIANS; i++) {
		visi_g[i].compute_sigma();
		totN += visi_g[i].n;
	}

	for (int i=0; i<NB_OCCL_GAUSSIANS; i++) {
		occl_g[i].compute_sigma();
		totN += occl_g[i].n;
	}

	totN += uniform_resp;
	weights[NB_GAUSSIANS] = uniform_resp/totN;

	for (int i=0; i<NB_VISI_GAUSSIANS; i++) {
		weights[i] = visi_g[i].n / totN;
		assert(weights[i]>=0 && weights[i]<=1);
	}

	for (int i=0; i<NB_OCCL_GAUSSIANS; i++) {
		weights[NB_VISI_GAUSSIANS+i] = occl_g[i].n / totN;
		assert(weights[NB_VISI_GAUSSIANS+i]>=0 && weights[NB_VISI_GAUSSIANS+i]<=1);
	}


	recycle=true;
	iteration++;

	if (save_images) {
		sprintf(str, "proba%02d.png", iteration );
		IplImage probaStub = proba;
		scale_save(str, &probaStub);
	}
}

float EMVisi2::process_pixel(const float *rgb, const float *frgb, const float dl, const float nccv, const float ncch, float *proba, float *visi_proba)
{
	// store responsabilities for each gaussian
	float resp[NB_GAUSSIANS+1];
	float sum_resp=0;
	float *w = weights;
	float *r = resp;

	float epsilon = 1e-40;

	// E-step: compute expectation of latent variables
	for (int i=0; i<NB_VISI_GAUSSIANS; i++) {
		*r = *w++ * visi_g[i]._proba(frgb) * dl * nccv;

		assert(_finite(*r));
		if (*r<0) *r = 0;
		if (*r>(1-epsilon)) *r= 1-epsilon;
		assert(*r >=0);
		assert(*r <=1);
		sum_resp += *r;
		r++;
	}
	float sum_visi_resp = sum_resp;

	for (int i=0; i<NB_OCCL_GAUSSIANS; i++) {
		*r = *w++ * occl_g[i]._proba(rgb) * ncch;
		if (*r<epsilon) *r = epsilon;
		assert(_finite(*r));
		assert(*r >=0);
		assert(*r <1);
		sum_resp += *r;
		r++;
	}
	resp[NB_GAUSSIANS] = *w * ncch *  1.0f/(255*255*255);
	sum_resp += resp[NB_GAUSSIANS];

	float sum_resp_inv = 1.0f/sum_resp;
	for (int i=0; i<NB_GAUSSIANS+1; i++)
		resp[i] *= sum_resp_inv;

	// M-step: update means and covariance matrices
	for (int i=0; i<NB_VISI_GAUSSIANS; i++) 
		visi_g[i].accumulate(frgb, resp[i]);

	for (int i=0; i<NB_OCCL_GAUSSIANS; i++) 
		occl_g[i].accumulate(rgb, resp[NB_VISI_GAUSSIANS+i]);

	uniform_resp += resp[NB_GAUSSIANS];

	*proba = sum_visi_resp * sum_resp_inv;
	if (visi_proba) *visi_proba = sum_visi_resp;

	return sum_resp;
}

bool EMVisi2::init() {
	ncc_h.setHistogram(ncc_proba_h);
	if (!ncc_h.loadHistogram("ncc_proba_h.mat") && !ncc_h.loadHistogram("../ncc_proba_h.mat")) {
		cerr << "can't load ncc_proba_h.mat histogram. Using built in distribution.\n";
	}
	ncc_v.setHistogram(ncc_proba_v);
	if (!ncc_v.loadHistogram("ncc_proba_v.mat") && !ncc_v.loadHistogram("../ncc_proba_v.mat")) {
		cerr << "can't load ncc_proba_v.mat histogram. Using built in distribution.\n";
	}
	return true;
}

int EMVisi2::setModel(const IplImage *im1, const IplImage *mask)
{
	if (proba.empty()) {
		dL.create(Size(im1->width, im1->height), CV_32FC1);
		ncc.create(Size(im1->width, im1->height), CV_32FC1);
		sum.create(Size(im1->width, im1->height), CV_32FC1);
		proba.create(Size(im1->width, im1->height), CV_32FC1);
		visi_proba.create(Size(im1->width, im1->height), CV_32FC1);
		nccproba_v.create(Size(im1->width, im1->height), CV_32FC1);
		nccproba_h.create(Size(im1->width, im1->height), CV_32FC1);
		if(im1->nChannels == 1){
			ratio.create(Size(im1->width, im1->height), CV_32FC1);
			im1f.create(Size(im1->width, im1->height), CV_32FC1);
		}else if(im1->nChannels == 3){
			ratio.create(Size(im1->width, im1->height), CV_32FC3);
			im1f.create(Size(im1->width, im1->height), CV_32FC3);
		}else{
			assert( 0 && "not supported channels");
		}
	}

	if (im1->nChannels >1) {
		IplImage *green1 = cvCreateImage(cvGetSize(im1), im1->depth, 1);
		cvSplit(im1, 0, green1, 0,0);
		fncc.setModel(green1, mask);
		cvReleaseImage(&green1);
	} else {
		fncc.setModel(im1, mask);
	}

	IplImage im1fStub = im1f;
	cvCvtScale(im1,&im1fStub);

	if (mask) 
		this->mask = Mat(mask);
	return 0;
}

int EMVisi2::setTarget(const IplImage *target)
{
	assert(im1f.cols == target->width && im1f.rows==target->height);
	iteration=0;

	if (target->depth != IPL_DEPTH_32F) {
		Mat targetStub(target, true);
		targetStub.convertTo(_im2, CV_32F);
	}

	assert(!ncc.empty());
	IplImage *green2;
	if (_im2.channels()>1) {
		green2 = cvCreateImage(cvGetSize(target), target->depth, 1);
		cvSplit(target, 0, green2, 0,0);
		fncc.setImage(green2);
		IplImage nccStub = ncc;
		IplImage sumStub = sum;
		fncc.computeNcc(ncc_size, &nccStub, sum.empty() ? 0 : &sumStub);
	} else {
		//green2.attach(const_cast<IplImage *>(im2),false);
		green2 = cvCloneImage(target);
		IplImage nccStub = ncc;
		IplImage sumStub = sum;
		fncc.computeNcc(ncc_size, &nccStub, sum.empty() ? 0 : &sumStub);
	}

	if (save_images) {
		IplImage nccStub = ncc;
		IplImage sumStub = sum;
		scale_save("ncc.png", &nccStub);
		scale_save("ncc_tex.png", &sumStub);
	}

#pragma omp parallel sections
	{
		IplImage nccStub = ncc;
		IplImage sumStub = sum;
		IplImage nccproba_vStub = nccproba_v;
		IplImage nccproba_hStub = nccproba_h;
#pragma omp section
		ncc_v.getProba(&nccStub, &sumStub, &nccproba_vStub);
#pragma omp section
		ncc_h.getProba(&nccStub, &sumStub, &nccproba_hStub);
	}
	if (save_images) {
		IplImage nccproba_vStub = nccproba_v;
		IplImage nccproba_hStub = nccproba_h;
		save_proba("nccproba_v.png", &nccproba_vStub);
		save_proba("nccproba_h.png", &nccproba_hStub);
	}

	{

		static float table[256][256];
		static float dtable[256][256];
		static bool table_computed=false;
		if (!table_computed) {
			table_computed=true;
			for (int i=0;i<256;i++) {
				for (int j=0;j<256;j++) {
					if (i==0 && j==0) { 
						table[i][j]=0;
						dtable[i][j]=1e-10;
					} else {
						table[i][j] = (180.0/CV_PI)*atan2((double)i+1,(double)j+1);
						dtable[i][j] = (180.0/CV_PI)/((i+1) + (1 + (j+1)*(j+1)/(i+1)/(i+1)));

						// this also works
						/*
						float s = 64;
						table[i][j] = 45*(j+s)/(i+s);
						dtable[i][j] = 45.0/(i+s);
						*/
					}
				}
			}
		}
		int n=im1f.cols*im1f.channels();
		for (int y=0;y<im1f.rows;y++) {
			float *a = (float *)im1f.ptr(y); 
			//float *b = &CV_IMAGE_ELEM(im2, float, y, 0);
			float *b = (float*) _im2.ptr(y);
			float *d = (float *)ratio.ptr(y);
			float *dl = (float *)dL.ptr(y);
			for (int x=0;x<n; x+=3) {
				int ia[3];
				int ib[3];
				for (int j=0; j<3; j++) {
					ia[j] = cvRound(a[x+j]);
					ib[j] = cvRound(b[x+j]);
					if (ia[j]<0) ia[j]=0;
					if (ib[j]<0) ib[j]=0;
					if (ia[j]>255) ia[j]=255;
					if (ib[j]>255) ib[j]=255;

					d[x+j] = table[ia[j]][ib[j]];
				}
				dl[x/3] = dtable[ia[0]][ib[0]]*dtable[ia[1]][ib[1]]*dtable[ia[2]][ib[2]];
				assert(dl[x/3]>0);
			}
		}
		if (save_images){
			IplImage dLStub = dL;
			scale_save("dL.png", &dLStub);
		}

		if (save_images) {
			IplImage ratioStub = ratio;
			scale_save("ratio.png", &ratioStub);
		}
	}

	return 0;
}

#ifdef WITH_GRAPHCUT

#include <vector>
#include "graph.h"
#include "graph.hpp"
#include "maxflow.hpp"

using namespace std;
typedef Graph<float, float, float> FGraph;

/*!
Tags connected '0' regions with an id (1-254)
return total area
*/
static double connected_regions(IplImage *mask, vector<CvConnectedComp> &regions)
{
	assert(mask->nChannels == 1);
	assert(mask->depth == IPL_DEPTH_8U);

	regions.clear();
	regions.reserve(254);

	int region = 1;
	double area = 0;

	for (int y=0; y<mask->height; y++) {
		unsigned char *m = &CV_IMAGE_ELEM(mask, unsigned char, y, 0);

		for (int x=0; x<mask->width; x++) {
			if (m[x]==0) {
				CvConnectedComp conn;

				cvFloodFill(mask, cvPoint(x,y), cvScalarAll(region), cvScalarAll(0), cvScalarAll(0),
					&conn, 8+CV_FLOODFILL_FIXED_RANGE );
				area += conn.area;
				conn.value.val[0] = region;
				regions.push_back(conn);

				region++;
				if (region==255) region=1;
			}
		}
	}
	return area;
}

static void display_err(char *e)
{
	cerr << "graph error: " << e << endl;
}


void EMVisi2::smooth(float amount, float threshold) {
	const IplImage *wa = proba;

	// Threshold proba image
	cv::Mat gc_mask(cvGetSize(proba), IPL_DEPTH_8U, 1);
	cvSet(gc_mask, cvScalarAll(255));

	// find pixels on which graph-cut should be applied
	for (int y=1; y<proba.height()-1; y++) {
		float *p = (float *) proba.ptr(y); 
		unsigned char *m = (unsigned char*) gc_mask.ptr(y);
		unsigned char *im = 0;
		if (mask.is_valid())
			im = mask.ptr(y);

		for (int x=1;x<proba.width()-1; x++)
			if ((im==0 || im[x]) // within mask and..
				&& (((p[x]>threshold) && (p[x] < (1-threshold))) // not very confident..
				|| ( fabs(p[x-1]-p[x])>.3) || (fabs(p[x-proba.step()]-p[x])>.3)	// transition
				)) {
					m[x]=0;
					/*
					if (x>0) m[x-1]=0;
					if (x<proba->width-1) m[x+1]=0;
					if (y<proba->height-1) m[x+mask.step()]=0;
					if (y>0) m[x-mask.step()]=0;
					*/
					m[x-1]=0;
					m[x+1]=0;
					m[x+mask.step()]=0;
					m[x-mask.step()]=0;

					// diag
					m[x+mask.step()+1]=0;
					m[x-mask.step()+1]=0;
					m[x+mask.step()-1]=0;
					m[x-mask.step()-1]=0;
			}
	}

	// segment connected uncertain areas
	vector<CvConnectedComp> regions;
	connected_regions(gc_mask, regions);
	if (save_images) {
		cv::Mat imreg(cvGetSize(gc_mask), IPL_DEPTH_8U, 3);

		CvMat *lut = cvCreateMat(1,256, CV_8UC3);
		CvRNG rng = cvRNG();
		cvRandArr(&rng, lut, CV_RAND_UNI, cvScalarAll(0), cvScalarAll(255));
		unsigned char *c = lut->data.ptr;
		//c[0] = c[1] = c[2] = 0;
		c[255*3] = c[255*3+1] = c[255*3+2] = 0;
		for (int y=0; y<imreg.height(); y++) {
			unsigned char *dst = imreg.ptr(y);
			unsigned char *src = gc_mask.ptr(y);
			for (int x=0; x<imreg.width(); x++) 
				for (int i=0; i<3; i++) 
					dst[x*3+i] = c[src[x]*3+i];
		}
		cvReleaseMat(&lut);
		cvSaveImage("regions.png", imreg);
	}

	// allocate the graph. Note: worst case memory scenario.
	int n_nodes= gc_mask.width()*gc_mask.height();
	int n_edges = 2*((wa->width)*(wa->height-1) + (wa->width-1)*wa->height);
	FGraph *g = new FGraph(n_nodes, n_edges, display_err);
	int *nodesid = new int[n_nodes];

	// try to run graphcut on all regions
	for (unsigned i=0; i<regions.size(); i++) {

		CvConnectedComp &r = regions[i];

		/*
		cout << "Region " << i << ": area=" << r.area << ", " 
		<< r.rect.width << "x" << r.rect.height << endl;
		*/

		g->reset();
		//g->add_node((int)r.area);
		g->add_node(r.rect.width * r.rect.height);
		for (int i=0; i<r.rect.width+1;i++) nodesid[i]=-1;

		int next_node = 0;

		unsigned rval = (unsigned)r.value.val[0];

		for (int y=r.rect.y; y<r.rect.y+r.rect.height; y++) {
			unsigned char *m = (unsigned char*) gc_mask.ptr(y);
			int *row_id = nodesid + (1+y-r.rect.y)*(r.rect.width+1)+1;
			row_id[-1]=-1;

			const float c = amount;

			float *proba_l = (float *)proba.ptr(y);
			float *visi_proba_l = (float *)visi_proba.ptr(y);

			for (int x=r.rect.x; x<r.rect.x+r.rect.width; x++) {
				if (m[x] == rval) {
					// add a new node
					*row_id = next_node;

					// terminal weights
					float wap = proba_l[x];
					float vp = visi_proba_l[x];
					g->add_tweights(next_node, 
						//-logf(PF*wap), -logf((1-PF)*(1-wap)));
						-log(PF*vp), -log((1-PF)*(vp/wap - vp)));

					// fill connectivity edges ..

					// .. up ?
					int up_id = row_id[-(r.rect.width+1)]; 
					if (up_id>=0) {
						// the node exists. Link it.
						g->add_edge(next_node, up_id, c, c);
					}

					// .. left ?
					int left_id = row_id[-1];
					if (left_id >= 0) {
						// the node exists. Link it.
						g->add_edge(next_node, left_id, c, c);
					}

					// .. up+left ?
					int upleft_id = row_id[-(r.rect.width+1)-1];
					if (upleft_id >= 0) {
						// the node exists. Link it.
						g->add_edge(next_node, upleft_id, c, c);
					}

					// .. up+right ?
					int upright_id = row_id[-(r.rect.width+1)+1];
					if (upright_id >= 0) {
						// the node exists. Link it.
						g->add_edge(next_node, upright_id, c, c);
					}

					next_node++;
				} else {
					*row_id = -1;
				}
				row_id++;
			}
		}

		// solve maxflow
		g->maxflow();

		// write result back
		for (int y=r.rect.y; y<r.rect.y+r.rect.height; y++) {
			float *p = (float *)proba.ptr(y);
			int *row_id = nodesid + (1+y-r.rect.y)*(r.rect.width+1)+1;

			for (int x=r.rect.x; x<r.rect.x+r.rect.width; x++) {
				if (*row_id >= 0) {
					p[x] = (g->what_segment(*row_id) == FGraph::SOURCE ? 0 : 1);
				}
				row_id++;
			}
		}
	}
	delete[] nodesid;
	delete g;
}
#else
void EMVisi2::smooth(float, float) {
}
#endif