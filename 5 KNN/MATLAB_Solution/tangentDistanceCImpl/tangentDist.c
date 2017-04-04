#include "mex.h"
#include "td.h"
/* mex tangentDist.c  ortho.c td.c */
/*
Protypte tangentDist(img1, img2, height, width, choice, bkgrd_color)
img1 and img2 are column vectors, other parameters are scalers
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    double dist = 0.0; // result
    double *img1, *img2; //Input vectors
	int height, width;
	int *choice;
	double background;
	
    if (nrhs != 6)
        mexErrMsgTxt("Six inputs required.");
    if (nlhs != 1)
        mexErrMsgTxt("One output required.");
    if (mxGetN(prhs[4]) !=9)
		mexErrMsgTxt("Transformation choice vector must have 9 elements!");
	
    img1       = mxGetPr    (prhs[0]); //pointer to img1
    img2       = mxGetPr    (prhs[1]); //pointer to img2
	height     = (int)mxGetScalar(prhs[2]);
	width      = (int)mxGetScalar(prhs[3]);
	choice     = mxGetPr    (prhs[4]); // pointer to choice vector
	background = mxGetScalar(prhs[5]);
	
    dist=twoSidedTangentDistance(img1, img2, height, width, choice, background);
    plhs[0] = mxCreateDoubleScalar(dist);
}