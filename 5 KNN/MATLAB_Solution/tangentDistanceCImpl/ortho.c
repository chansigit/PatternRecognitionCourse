/*
    ortho.c -- implementation for orthonormalization routines
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
#include <math.h>
#include <float.h>

const double ortho_singular_threshold = 1e-9;
// if vector norms are smaller than this value, it is assumed that no
// orthonormal basis exists; DBL_MIN is way to small for this; 
// surely there exist better algorithms than this one w.r.t
// numerical stability

int orthonormalize (double **A, const unsigned int num, const unsigned int dim)
     // calculates an orthonormal basis using Gram-Schmidt
     // returns 0 if basis can be found, 1 otherwise
{
  unsigned int n,m,d;
  int retval=0;
  double projection,norm,tmp;
  double *A_n, *A_m;

  for (n=0; n<num; ++n) {
    A_n=(double*)A[n];
    for (m=0; m<n; ++m) {
      A_m=(double*)A[m];
      projection=0.0;
      for (d=0; d<dim; ++d) {
	// get projection onto existing vector (scalar product)
	projection+=A_n[d]*A_m[d];}
      for (d=0; d<dim; ++d) {
	// subtract component within existing subspace
	A_n[d]-=projection*A_m[d];}
    }
    // normalize
    norm=0.0;
    for (d=0; d<dim; ++d) {
      tmp=A_n[d];
      norm+=tmp*tmp;}
    if (norm<ortho_singular_threshold) {
      retval=1;}
    norm=1.0/sqrt(norm);
    for (d=0; d<dim; ++d) {
      A_n[d]*=norm;}
  }

  return retval;
} 

int orthonormalizeP (double **A, const unsigned int num, const unsigned int dim)
     // calculates an orthonormal basis using Gram-Schmidt
     // returns 0 if basis can be found, 1 otherwise
     // try parallelization for CPU architectures with multiple FPUs
{
  unsigned int n,m,d,dim1;
  int retval=0;
  double projection,norm,tmp;
  double projection1,projection2,projection3,projection4;
  double *A_n, *A_m;

  dim1=dim-dim%4;

  for (n=0; n<num; ++n) {
    A_n=(double*)A[n];
    for (m=0; m<n; ++m) {
      A_m=(double*)A[m];
      projection=0.0;
      projection1=0.0;
      projection2=0.0;
      projection3=0.0;
      projection4=0.0;
      for (d=0; d<dim1; d+=4) {
	projection1+=A_n[d]*A_m[d]; 
	projection2+=A_n[d+1]*A_m[d+1]; 
	projection3+=A_n[d+2]*A_m[d+2]; 
	projection4+=A_n[d+3]*A_m[d+3]; }
      projection=projection1+projection2+projection3+projection4;
      for (; d<dim; ++d) {
	projection+=A_n[d]*A_m[d];}
      for (d=0; d<dim1; d+=4) {
	A_n[d]-=projection*A_m[d];
	A_n[d+1]-=projection*A_m[d+1];
	A_n[d+2]-=projection*A_m[d+2];
	A_n[d+3]-=projection*A_m[d+3];}
      for (; d<dim; ++d) {
	A_n[d]-=projection*A_m[d];}
    }
    // normalize
    norm=0.0;
    for (d=0; d<dim; ++d) {
      tmp=A_n[d];
      norm+=tmp*tmp;}
    if (norm<ortho_singular_threshold) {
      retval=1;}
    norm=1.0/sqrt(norm);
    for (d=0; d<dim; ++d) {
      A_n[d]*=norm;}
  }

  return retval;
} 


int orthonormalizePzero (double **A, const unsigned int num, const unsigned int dim)
     // calculates an orthonormal basis using Gram-Schmidt
     // returns zero
     // sets tangents to zero, that are not "orthogonal enough" 
     // try parallelization for CPU architectures with multiple FPUs
{
  unsigned int n,m,d,dim1;
  double projection,norm,tmp;
  double projection1,projection2,projection3,projection4;
  double *A_n, *A_m;

  dim1=dim-dim%4;

  for (n=0; n<num; ++n) {
    A_n=(double*)A[n];
    for (m=0; m<n; ++m) {
      A_m=(double*)A[m];
      projection=0.0;
      projection1=0.0;
      projection2=0.0;
      projection3=0.0;
      projection4=0.0;
      for (d=0; d<dim1; d+=4) {
	projection1+=A_n[d]*A_m[d]; 
	projection2+=A_n[d+1]*A_m[d+1]; 
	projection3+=A_n[d+2]*A_m[d+2]; 
	projection4+=A_n[d+3]*A_m[d+3]; }
      projection=projection1+projection2+projection3+projection4;
      for (; d<dim; ++d) {
	projection+=A_n[d]*A_m[d];}
      for (d=0; d<dim1; d+=4) {
	A_n[d]-=projection*A_m[d];
	A_n[d+1]-=projection*A_m[d+1];
	A_n[d+2]-=projection*A_m[d+2];
	A_n[d+3]-=projection*A_m[d+3];}
      for (; d<dim; ++d) {
	A_n[d]-=projection*A_m[d];}
    }
    // normalize
    norm=0.0;
    for (d=0; d<dim; ++d) {
      tmp=A_n[d];
      norm+=tmp*tmp;}
    if (norm<ortho_singular_threshold) {
      norm=0.0;}
    else {
      norm=1.0/sqrt(norm);}
    for (d=0; d<dim; ++d) {
      A_n[d]*=norm;}
  }

  return 0;
} 
