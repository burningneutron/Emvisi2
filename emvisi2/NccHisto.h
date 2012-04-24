#ifndef __NCC_HISTO_H__
#define __NCC_HISTO_H__

#include <opencv2/opencv.hpp>

/* This class is used to represent an histogram of textureness/correlation 
 * on the background or on the foreground.
 */
class NccHisto {
public:
	NccHisto();
	~NccHisto();
	bool loadHistogram(const char *filename = "proba.mat");
	bool saveHistogram(const char *filename);
	void setHistogram(const float *histo);
	void getProba(const IplImage *ncc, IplImage *proba);
	void getProba(const IplImage *ncc, const IplImage *sumstdev, IplImage *proba);

	void initEmpty();
	void addElem(float corr, float var, float w)
	{
		lut[lut_idx(corr, var)] += w;
		nelem+=w;
	}
	void normalize(float bias);
	
	// these contain the number of bins for the NCC histograms.
	// the values 15,15 work for the distributed ones.
	// if you change these values, you have to rebuild the histograms.
	static const int NTEX=15;
	static const int NCORR=15;

	int tex_idx(float var) {
		int i = cvFloor(sqrtf(var));
		if (i>NTEX) return NTEX;
		if (i<0) return 0;
		return i;
	}

	int correl_idx(float ncc) {
		int i = cvFloor((ncc)*(NCORR+1)/1.0f);
		if (i>NCORR) return NCORR;
		if (i<0) return 0;
		return i;
	}

	int lut_idx(float ncc, float var) {
		return correl_idx(ncc)*(NTEX+1) + tex_idx(var);
	}

	float *lut;
	float *deleteme;
public:
	float nelem;
};

#endif