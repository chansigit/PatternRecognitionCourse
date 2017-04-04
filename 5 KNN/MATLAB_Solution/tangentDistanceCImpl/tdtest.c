/*
    tdtest.c -- testing the tangent distance routines
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

#include <stdio.h>
#include <pgm.h>
#include <assert.h>
#include <stdlib.h>
#include "td.h"


/** reads PGM image and returns allocated one-dimensional double array and height/width.
    make sure to free memory in calling routine */
double * readPGMimage(const char * name, int * height, int * width){
  gray ** pgmImage;
  FILE * file;
  gray maxval;
  double * image;
  int ind,i,j;

  file=fopen(name,"r");
  assert(file);

  pgmImage=pgm_readpgm(file,width,height,&maxval); 

  image=(double *)malloc((*height)*(*width)*sizeof(double));
  assert(image);

  ind=0;
  for(i=0;i<(*height);i++) 
    for(j=0;j<(*width);j++)
      image[ind++]=(double)pgmImage[i][j];

  fclose(file);
  
  pgm_freearray(pgmImage,(*height));

  return image;
}


int main(int argc, char** argv){

  int width,height,choice[]={1,1,1,1,1,1,0,0,0};
  double * imageOne, *imageTwo, dist, background=0.0;

  if(argc<3) {
    printf("usage: %s <infile1.pgm> <infile2.pgm>\n",argv[0]);
    exit(1);
  }

  imageOne=readPGMimage(argv[1], &height, &width);
  imageTwo=readPGMimage(argv[2], &height, &width);

  dist=tangentDistance(imageOne, imageTwo, height, width, choice, background);

  printf("Tangent distance between %s and %s is %g.\n",argv[1],argv[2],dist);

  dist=twoSidedTangentDistance(imageOne, imageTwo, height, width, choice, background);

  printf("Two-sided tangent distance between %s and %s is %g.\n",argv[1],argv[2],dist);

  free(imageOne);
  free(imageTwo);

  return 0;
}
