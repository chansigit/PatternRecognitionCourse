/*
    td.h -- header file for the tangent distance routines
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


/**
 * @file   td.h
 * @author Daniel Keysers & Thomas Deselaers
 * 
 * @brief functions to calculate tangent distance between two images.
 * 
 * More information about this can be found in 
 *
 *  D. Keysers. Approaches to Invariant Image Object
 *  Recognition. Diploma thesis, Lehrstuhl fuer Informatik VI, RWTH
 *  Aachen, Aachen, June 2000.
 * 
 */

#ifndef __td
#define __td

/* definitions for the implicit smoothing template*/
#define templatefactor1 0.1667
#define templatefactor2 0.6667
#define templatefactor3 0.08
/* the template looks like this (extended Sobel):
   0   f1 0 -f1  0
   f3  f2 0 -f2 -f3
   0   f1 0 -f1  0  */
/* note: this template works well for 16x16 sized images, for different sizes an additional smoothing
   might be helpful */


/* constant for brightness tangent */
#define additiveBrightnessValue 0.1

/* constant size of "choice" = max number of tangents */
#define maxNumTangents 9


/** 
 * Two dimensional access on images saved in one dimensional array
 * 
 * @param y the y coordinate to be accessed
 * @param x the x coordinate to be accessed
 * @param width the assumed width of the image
 * 
 * @return the coordinate where this (y,x)-point can be found
 */
inline int tdIndex(int y, int x, int width);


/** 
 * Calculate selected tangents for an image.
 * 
 * @param image the input image. The tangents will be calculated for
 * this image 

 * @param tangents memory space where the tangents are to be
 * saved. This memory has to be allocated in advance by the user and
 * has to be freed by the user, too.

 * @param numTangents the number of tangents that shall be
 * created. Has to be consistent with choice.

 * @param height The height of the image.
 * @param width The width of the image.

 * @param choice select which tangents you want to get. 
 * set choice[i] = 1 to get the tangent, choice[i]=0 to skip it
 * i=0 - horizontal shift
 * i=1 - vertical shift
 * i=2 - hyperbolic 1
 * i=3 - hyperbolic 2
 * i=4 - scaling
 * i=5 - rotation
 * i=6 - line thickness
 * i=7 - additive brightness
 * i=8 - multiplicative brightness
 *
 * @param background the color to be assumed around the image
 * 
 * @return the number of tangents created
 */
int calculateTangents(const double * image, double ** tangents, const int numTangents,
                      const int height, const int width, const int * choice, const double background);

/** 
 * Finds an orthonormal basis for a given set of tangents
 * 
 * @param tangents The tangents to be orthonormalized.
 * @param numTangents The number of tangents to be found.
 * @param height The height of the tangents.
 * @param width The width of the tangents.
 */
int normalizeTangents(double ** tangents, const int numTangents, const int height, const int width);

/** Calculates the distance between two images given a set of orthonormalized tangents. 
 *
 * @param imageOne reference image
 * @param imageTwo test image
 * imageOne and imageTwo have to be the same size.
 * @param tangents A set of orthonormalized tangents. e.g. created by
 * calculateTangents and normalizeTangents.
 * @param numTangents The number of tangents in the tangent array.
 * @param height The height of the images and tangents.
 * @param width The width of the images and tangents
 * 
 * @return the distance between the two images.
 */
double calculateDistance(const double * imageOne, const double * imageTwo, const double ** tangents,
                         const int numTangents, const int height, const int width);

/** Calculates the tangent distance between two images given as 1-D double arrays.  
 *  choice must have at least maxNumTangents elements 
 * 
 * This function is a wrapper function for calculateTangents, normalizeTangents and calculateDistance. 
 *
 * @param imageOne Reference image.
 * @param imageTwo Test image.
 * @param height The height of the images
 * @param width The width of the images.
 * @param choice select which tangents you want to get. 
 * set choice[i] = 1 to get the tangent, choice[i]=0 to skip it
 * i=0 - horizontal shift
 * i=1 - vertical shift
 * i=2 - hyperbolic 1
 * i=3 - hyperbolic 2
 * i=4 - scaling
 * i=5 - rotation
 * i=6 - line thickness
 * i=7 - additive brightness
 * i=8 - multiplicative brightness
 *
 * @param background Color to be assumed around the image.
 * 
 * @return The distance between imageOne and imageTwo.
 */
double tangentDistance(const double * imageOne, const double * imageTwo, 
                       const int height, const int width, const int * choice, const double background);


/** Calculates the two-sided tangent distance between two images given as 1-D double arrays.  
 * choice must have at least maxNumTangents elements 
 * 
 * This function is a wrapper function for calculateTangents, normalizeTangents and calculateDistance. 
 *
 * @param imageOne Reference image.
 * @param imageTwo Test image.
 * @param height The height of the images
 * @param width The width of the images.
 * @param choice select which tangents you want to get. 
 * set choice[i] = 1 to get the tangent, choice[i]=0 to skip it
 * i=0 - horizontal shift
 * i=1 - vertical shift
 * i=2 - hyperbolic 1
 * i=3 - hyperbolic 2
 * i=4 - scaling
 * i=5 - rotation
 * i=6 - line thickness
 * i=7 - additive brightness
 * i=8 - multiplicative brightness
 *
 * @param background Color to be assumed around the image.
 * 
 * @return The distance between imageOne and imageTwo.
 */
double twoSidedTangentDistance(const double * imageOne, const double * imageTwo, 
                               const int height, const int width, int * choice, const double background);

/** This returns the closest image/point within the tangent subspace.
 */
void calculateClosest(const double * basePoint, const double * testPoint,
		      double ** tangents, const int numTangents, 
		      const int size, double * closest);

/** This returns a vector from the point/image to the closest point/image within the tangent subspace.
 */
void calculatePerpendicular(const double * basePoint, const double * testPoint,
                            double ** tangents, const int numTangents, 
                            const int height, const int width,
                            double * perpendicular);

#endif
