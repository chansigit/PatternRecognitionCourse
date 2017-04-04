/*
    td.c -- implementation of the tangent distance routines
    Copyright (C) 2003 Daniel Keysers

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/
#include <assert.h>
#include <stdlib.h>
#include "td.h"
#include "ortho.h"

// Two dimensional access on images saved in one dimensional array
inline int tdIndex(int y, int x, int width){
  return y*width+x;
}


/** Calculates the tangents for one image. */
int calculateTangents(const double * image, double ** tangents, const int numTangents,
                      const int height, const int width, const int * choice, const double background){
  int j,k,ind,tangentIndex,maxdim;
  double tp,factorW,offsetW,factorH,factor,offsetH,halfbg;
  double *tmp, *x1, *x2, *currentTangent;
  
  int size=height*width;
  maxdim=(height>width)?height:width;

  tmp=(double*)malloc(maxdim*sizeof(double));
  x1=(double*)malloc(size*sizeof(double));
  x2=(double*)malloc(size*sizeof(double));
  // no assertions on memory allocation here because of the Linux memory management policy
  // if it makes sense, insert the following here:
  // assert(tmp); assert(x1); assert(x2);
      
  factorW=((double)width*0.5);
  offsetW=0.5-factorW;
  factorW=1.0/factorW;

  factorH=((double)height*0.5);
  offsetH=0.5-factorH;
  factorH=1.0/factorH;

  factor=(factorH<factorW)?factorH:factorW; //min

  halfbg=0.5*background;


  /* x1 shift along width */
  /* first use mask 1 0 -1 */
  for(k=0; k<height; k++) {
    /* first column */
    ind=tdIndex(k,0,width);
    x1[ind]= halfbg - image[ind+1]*0.5;
    /* other columns */
    for(j=1; j<width-1;j++) {
      ind=tdIndex(k,j,width);
      x1[ind]=(image[ind-1]-image[ind+1])*0.5;
    }
    /* last column */
    ind=tdIndex(k,width-1,width);
    x1[ind]= image[ind-1]*0.5 - halfbg;
  }
  /* now compute 3x3 template */
  /* first line */
  for(j=0;j<width;j++) {
    tmp[j]=x1[j];
    x1[j]=templatefactor2*x1[j]+templatefactor1*x1[j+width];
  }
  /* other lines */
  for(k=1;k<height-1;k++)
    for(j=0;j<width;j++) {
      ind=tdIndex(k,j,width);
      tp=x1[ind];
      x1[ind]=templatefactor1*tmp[j]+templatefactor2*x1[ind]+
	templatefactor1*x1[ind+width];
      tmp[j]=tp;
    }
  /* last line */
  for(j=0;j<width;j++) {
    ind=tdIndex(height-1,j,width);
    x1[ind]=templatefactor1*tmp[j]+templatefactor2*x1[ind];
  }
  /* now add the remaining parts outside the 3x3 template */
  /* first two columns */
  for(j=0;j<2;j++)
    for(k=0;k<height;k++) {
      ind=tdIndex(k,j,width);
      x1[ind]+=templatefactor3*background;
    } 
  /* other columns */
  for(j=2;j<width;j++)
    for(k=0;k<height;k++) {
      ind=tdIndex(k,j,width);
      x1[ind]+=templatefactor3*image[ind-2];
    } 
  for(j=0;j<width-2;j++)
    for(k=0;k<height;k++) {
      ind=tdIndex(k,j,width);
      x1[ind]-=templatefactor3*image[ind+2];
    }
  /* last two columns*/
  for(j=width-2;j<width;j++)
    for(k=0;k<height;k++) {
      ind=tdIndex(k,j,width);
      x1[ind]-=templatefactor3*background;
    }


  /*x2 shift along height */
  /* first use mask 1 0 -1 */
  for(j=0; j<width;j++) {
    /* first line */
    x2[j]= halfbg - image[j+width]*0.5;
    /* other lines */
    for(k=1; k<height-1; k++) {
      ind=tdIndex(k,j,width);
      x2[ind]=(image[ind-width]-image[ind+width])*0.5;
    }
    /* last line */
    ind=tdIndex(height-1,j,width);
    x2[ind]= image[ind-width]*0.5 - halfbg;
  }

  /* now compute 3x3 template */
  /* first column */
  for(j=0;j<height;j++) {
    ind=tdIndex(j,0,width);
    tmp[j]=x2[ind];
    x2[ind]=templatefactor2*x2[ind]+templatefactor1*x2[ind+1];
  }
  /* other columns */
  for(k=1;k<width-1;k++)
    for(j=0;j<height;j++) {
      ind=tdIndex(j,k,width);
      tp=x2[ind];
      x2[ind]=templatefactor1*tmp[j]+templatefactor2*x2[ind]+
	templatefactor1*x2[ind+1];
      tmp[j]=tp;
    }
  /* last column */
  for(j=0;j<height;j++) {
    ind=tdIndex(j,width-1,width);
    x2[ind]=templatefactor1*tmp[j]+templatefactor2*x2[ind];
  }

  /* now add the remaining parts outside the 3x3 template */
  for(j=0;j<2;j++)
    for(k=0;k<width;k++) {
      ind=tdIndex(j,k,width);
      x2[ind]+=templatefactor3*background;
    } 
  for(j=2;j<height;j++)
    for(k=0;k<width;k++) {
      ind=tdIndex(j,k,width);
      x2[ind]+=templatefactor3*image[ind-2*width];
    } 
  for(j=0;j<height-2;j++)
    for(k=0;k<width;k++) {
      ind=tdIndex(j,k,width);
      x2[ind]-=templatefactor3*image[ind+2*width];
    }
  for(j=height-2;j<height;j++)
    for(k=0;k<width;k++) {
      ind=tdIndex(j,k,width);
      x2[ind]-=templatefactor3*background;
    }


  /* now go through the tangents */

  tangentIndex=0;

  if(choice[0]>0){  // horizontal shift
    currentTangent=tangents[tangentIndex];
    for(ind=0;ind<size;ind++) currentTangent[ind]=x1[ind];
    tangentIndex++;
  }

  if(choice[1]>0){  // vertical shift
    currentTangent=tangents[tangentIndex];
    for(ind=0;ind<size;ind++) currentTangent[ind]=x2[ind];
    tangentIndex++;
  }

  if(choice[2]>0){  // hyperbolic  1
    //    Vapnik book says this is "diagonal deformation" (error), 
    //    this is the "axis deformation" 
    currentTangent=tangents[tangentIndex];
    ind=0;
    for(k=0;k<height;k++)
      for(j=0;j<width;j++) {
	currentTangent[ind] = ((j+offsetW)*x1[ind] - (k+offsetH)*x2[ind])*factor;
	ind++;
      }
    tangentIndex++;
  }

  if(choice[3]>0){  // hyperbolic  2, (description = inverse of hyperbolic 1)
    currentTangent=tangents[tangentIndex];
    ind=0;
    for(k=0;k<height;k++)  
      for(j=0;j<width;j++) {
	currentTangent[ind] = ((k+offsetH)*x1[ind] + (j+offsetW)*x2[ind])*factor;
	ind++;
      }
    tangentIndex++;
  }

  if(choice[4]>0){  // scaling
    currentTangent=tangents[tangentIndex];
    ind=0;
    for(k=0;k<height;k++)
      for(j=0;j<width;j++) {
	currentTangent[ind] = ((j+offsetW)*x1[ind] + (k+offsetH)*x2[ind])*factor;
	ind++;
      }
    tangentIndex++;
  }

  if(choice[5]>0){  // rotation
    currentTangent=tangents[tangentIndex];
    ind=0;
    for(k=0;k<height;k++)
      for(j=0;j<width;j++) {
	currentTangent[ind] = ((k+offsetH)*x1[ind] - (j+offsetW)*x2[ind])*factor;
	ind++;
      }
    tangentIndex++;
  }

  if(choice[6]>0){  // line thickness
    currentTangent=tangents[tangentIndex];
    ind=0;
    for(k=0;k<height;k++)
      for(j=0;j<width;j++) {
	currentTangent[ind] = x1[ind]*x1[ind] + x2[ind]*x2[ind];
	ind++;
      } 
    tangentIndex++;
  }

  if(choice[7]>0){  // additive brightness
    currentTangent=tangents[tangentIndex];
    for(ind=0;ind<size;ind++)
      currentTangent[ind] = additiveBrightnessValue; 
    tangentIndex++;
  }

  if(choice[8]>0){  // multiplicative brightness
    currentTangent=tangents[tangentIndex];
    for(ind=0;ind<size;ind++)
      currentTangent[ind] = image[ind];
    tangentIndex++;
  }

  free(tmp);
  free(x1);
  free(x2);
  
  assert(tangentIndex==numTangents);

  return tangentIndex;
}

/** Finds an orthonormal basis for a given set of tangents using Gram-Schmidt orthogonalization. */
int normalizeTangents(double ** tangents, const int numTangents, const int height, const int width){

  unsigned int size=(unsigned int)height*width;
  
  orthonormalizePzero (tangents, (unsigned int) numTangents, size);

  // here we always return the original number of tangents dimensions
  // "lost" in the normalization are set to zero vectors this is not
  // what was intended originally, but it works and is kept for
  // backward compatibility
  return numTangents;
}


/** Calculates the distance between two images given a set of orthonormalized tangents. */
double calculateDistance(const double * imageOne, const double * imageTwo, const double ** tangents,
			 const int numTangents, const int height, const int width){

  double dist=0.0,tmp;
  const double *tangents_k;
  int k,l;

  int size=height*width;

  // first calculate squared Euclidean distance
  for(l=0;l<size;++l){
    tmp=imageOne[l]-imageTwo[l];
    dist+=tmp*tmp;
  }
  
  // then subtract the part within the subspace 
  for(k=0;k<numTangents;++k){
    tangents_k=tangents[k];
    tmp=0.0;
    for(l=0;l<size;++l) tmp+=(imageOne[l]-imageTwo[l])*tangents_k[l];
    dist-=tmp*tmp;
  }

  return dist;
}

/** Calculates the tangent distance between two images given as 1-D double arrays.  */
/** choice must have at least maxNumTangents elements */
double tangentDistance(const double * imageOne, const double * imageTwo, 
                       const int height, const int width, const int * choice, const double background){
  int i,numTangents=0,numTangentsRemaining;
  double ** tangents,dist;

  int size=width*height;

  for(i=0;i<maxNumTangents;++i) {
    if(choice[i]>0) numTangents++;
  }

  tangents=(double **)malloc(numTangents*sizeof(double *));
  // assert(tangents);
  for(i=0;i<numTangents;++i) {
    tangents[i]=(double *)malloc(size*sizeof(double));
    // assert(tangents[i]);
  }


  // determine the tangents of the first image
  calculateTangents(imageOne, tangents, numTangents, height, width, choice, background);

  // find the orthonormal tangent subspace 
  numTangentsRemaining = normalizeTangents(tangents, numTangents, height, width);

  // determine the distance to the closest point in the subspace
  dist=calculateDistance(imageOne, imageTwo, (const double **) tangents, numTangentsRemaining, height, width);


  for(i=0;i<numTangents;++i) {
    free(tangents[i]);
  }
  free(tangents);

  return dist;
}


/** Calculates the two-sided tangent distance between two images given as 1-D double arrays.  */
/** choice must have at least maxNumTangents elements */
double twoSidedTangentDistance(const double * imageOne, const double * imageTwo, 
                               const int height, const int width, int * choice, const double background){
  int i,numTangents=0,numTangentsRemaining;
  double ** tangents,dist;

  int size=width*height;

  for(i=0;i<maxNumTangents;++i) {
    if(choice[i]>0) numTangents++;
  }

  tangents=(double **)malloc(2*numTangents*sizeof(double *));
  // assert(tangents>0);
  for(i=0;i<2*numTangents;++i) {
    tangents[i]=(double *)malloc(size*sizeof(double));
    // assert(tangents[i]>0);
  }

  // determine the tangents of the images
  calculateTangents(imageOne, tangents,                 numTangents, height, width, choice, background);
  calculateTangents(imageTwo, &(tangents[numTangents]), numTangents, height, width, choice, background);

  // find the orthonormal tangent subspace 
  numTangentsRemaining = normalizeTangents(tangents, 2*numTangents, height, width);

  // determine the distance to the closest point in the subspace
  dist=calculateDistance(imageOne, imageTwo, (const double **) tangents, numTangentsRemaining, height, width);


  for(i=0;i<2*numTangents;++i) {
    free(tangents[i]);
  }
  free(tangents);

  return dist;
}

/** This returns the closest point in the tangent subspace, given as
 * orthonormal basis. This can be useful, if you want to calculate
 * other distances than the Euclidean.
 */
void calculateClosest(const double * basePoint, const double * testPoint,
		      double ** tangents, const int numTangents, 
		      const int size, double * closest) {
  int i,tan;
  double alpha, *tangent;

  // start with base point
  for(i = 0; i < size; ++i) closest[i] = basePoint[i];
  

  for(tan = 0; tan < numTangents; ++tan) {
    tangent = tangents[tan];
    alpha = 0.0;

    // determine projection onto tangent
    for(i = 0; i < size; ++i)
      alpha += (testPoint[i]-basePoint[i])*tangent[i];
    
    // add component in tangent direction
    for(i = 0; i < size; ++i)
      closest[i] += alpha * tangent[i];
  }
  
}


/** This returns a vector from the point to the closest point in the
 * tangent subspace, given as orthonormal basis. This can be useful,
 * if you want to calculate other distances than the Euclidean; the
 * (squared) length of this vector should be identical to the tangent
 * distance.
 */
void calculatePerpendicular(const double * basePoint, const double * testPoint,
                            double ** tangents, const int numTangents, 
                            const int height, const int width,
                            double * perpendicular) {
  int i;
  int size=width*height;
 
  calculateClosest(basePoint, testPoint, tangents, numTangents, size, perpendicular);
  
  for(i = 0; i < size; ++i)
    perpendicular[i] -= testPoint[i];

}

