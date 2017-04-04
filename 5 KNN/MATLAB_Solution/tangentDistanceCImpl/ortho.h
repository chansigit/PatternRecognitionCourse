/*
    ortho.h -- header file for orthonormalization routines
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
#ifndef __ortho
#define __ortho

int orthonormalize (double **A, const unsigned int num, const unsigned int dim);
     // calculates an orthonormal basis using Gram-Schmidt
     // returns 0 if basis can be found, 1 otherwise

int orthonormalizeP (double **A, const unsigned int num, const unsigned int dim);
     // calculates an orthonormal basis using Gram-Schmidt
     // returns 0 if basis can be found, 1 otherwise

int orthonormalizePzero (double **A, const unsigned int num, const unsigned int dim);
     // calculates an orthonormal basis using Gram-Schmidt
     // returns 0 
     // sets tangents to zero, that are not "orthogonal enough" 

#endif
