#include "NccHisto.h"
#include "growmat.h"

NccHisto::NccHisto() : lut(0), deleteme(0) {
}

NccHisto::~NccHisto() {
	if (deleteme) delete[] deleteme;
}

void NccHisto::setHistogram(const float *histo)
{
	int n =(NTEX+1)*(NCORR+1);
	lut = deleteme = new float[n];
	memcpy(lut, histo, sizeof(float)*n);
}

bool NccHisto::loadHistogram(const char *filename)
{
	CvGrowMat *histo = CvGrowMat::loadMat(filename, CV_32FC1);
	if (!histo) return false;
	if (histo->rows != (NCORR+1) || histo->cols!=(NTEX+1)) {
		std::cerr << filename << ": wrong matrix size.\n";
		return false;
	}
	float *_lut = new float[histo->rows*histo->cols];
	lut = _lut;
	if (deleteme) delete[] deleteme;
	deleteme = lut;

	for (int i=0;i<histo->rows;i++)
		for (int j=0;j<histo->cols;j++)
			_lut[i*(NTEX+1)+j] = cvGetReal2D(histo, i, j);

	delete histo;
	return true;
}

bool NccHisto::saveHistogram(const char *filename)
{
	CvMat m;
	cvInitMatHeader(&m, NCORR+1, NTEX+1, CV_32FC1, lut);
	return CvGrowMat::saveMat(&m, filename);
}

void NccHisto::getProba(const IplImage *ncc, IplImage *proba)
{
	if (lut==0) loadHistogram();
	assert(lut);
	if (lut==0) return;
	assert(ncc->nChannels==3);
	assert(ncc->width == proba->width && ncc->height==proba->height);
	assert(proba->nChannels==1);

	const int w=ncc->width;
	const int h=ncc->height;

	for (int y=0; y<h;y++) {
		float *dst = &CV_IMAGE_ELEM(proba,float,y,0);
		const float *src = &CV_IMAGE_ELEM(ncc,float,y,0);
		for (int x=0;x<w;x++) {
			dst[x] = lut[lut_idx(src[x*3], src[x*3+1])];
		}
	}
}

void NccHisto::getProba(const IplImage *ncc, const IplImage *sumstdev, IplImage *proba)
{
	if (lut==0) loadHistogram();
	assert(lut);
	if (lut==0) return;
	assert(ncc->nChannels==1);
	assert(sumstdev->nChannels==1);
	assert(ncc->width == proba->width && ncc->height==proba->height);
	assert(proba->nChannels==1);

	const int w=ncc->width;
	const int h=ncc->height;

	for (int y=0; y<h;y++) {
		float *dst = &CV_IMAGE_ELEM(proba,float,y,0);
		const float *src = &CV_IMAGE_ELEM(ncc,float,y,0);
		const float *sum = &CV_IMAGE_ELEM(sumstdev,float,y,0);
		for (int x=0;x<w;x++) {
			dst[x] = lut[lut_idx(src[x], sum[x])];
		}
	}
}

void NccHisto::initEmpty() {
	int n =(NTEX+1)*(NCORR+1);
	lut = deleteme = new float[n];
	for (int i=0; i<n; i++) lut[i] = 0.0f;
	nelem=0;
}

void NccHisto::normalize(float bias)
{
	int n =(NTEX+1)*(NCORR+1);
	float div = nelem + n*bias;
	for (int i=0; i<n; i++) lut[i] = (lut[i]+bias) / div;
}