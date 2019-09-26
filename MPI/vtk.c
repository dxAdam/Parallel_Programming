#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
function that plots the Temperature in a format suitable for paraview and VisIt
all the arguments are inputs and are constant as this fucntion only is for
visualization

Xmin : start of the computational domain in X-dirextion
Xmax : end of the computational domain in X-direction
Ymin : start of the computational domain in Y-direction
Ymax : end of the computational domain in Y-direction

N: number of points along the X-direction
M: number of points along the Y-direction

T: Temparature field specified as a 1D array as T[j*N+i] where 
0<=i<N,  0<=j<M 

index: index of the output file outINDEX.vtk
The 0<k<1 implies that 2D is a special case of 3D and hence the number of
discrete points in Z-direction is 1.

dont forget to LINK WITH MATH LIBRARY since we are using exp() function
*/


void VTK_out(const int N, const int M, const double *Xmin, const double *Xmax,
             const double *Ymin, const double *Ymax, const double *T,
             const int index) 
{
  // N is number of segments

  unsigned int i, j, k;

  double dx = ((*Xmax) - (*Xmin)) / (N - 1);
  double dy = ((*Ymax) - (*Ymin)) / (M - 1);

  FILE *fp = NULL;
  char filename[64];
  sprintf(filename, "out%d.vtk", index);
  fp = fopen(filename, "w");

  fprintf(fp, "# vtk DataFile Version 2.0 \n");
  fprintf(fp, "Grid\n");
  fprintf(fp, "ASCII\n");
  fprintf(fp, "DATASET STRUCTURED_GRID\n");
  fprintf(fp, "DIMENSIONS %d %d %d\n", N, M, 1);
  fprintf(fp, "POINTS %d float\n", (N) * (M) * (1));

  for (k = 0; k < 1; k++) {
    for (j = 0; j < M; j++) {
      for (i = 0; i < N; i++) {
        fprintf(fp, "%lf %lf %lf\n", (*Xmin) + i * dx, (*Ymin) + j * dy, 0.0);
      }
    }
  }
  fprintf(fp, "POINT_DATA %d\n", (N) * (M) * (1));
  fprintf(fp, "SCALARS P float 1\n");
  fprintf(fp, "LOOKUP_TABLE default\n");

  int cout = 0;

  for (k = 0; k < 1; k++) {
    for (j = 0; j < M; j++) {
      for (i = 0; i < N; i++) {
        fprintf(fp, " %lf\n", T[j * N + i]);
      }
    }
  }

  fclose(fp);
}

// as an example I used the exact function T=x*exp(y) with following parameters for N and M;

/*int main()
{

  double Xmin = 0.0;
  double Xmax = 2.0;
  double Ymin = 0.0;
  double Ymax = 1.0;

  int N = 101;
  int M = 201;

  double x, y;

// calculate delta_x and delta_y

  double dx = (Xmax - Xmin) / (N - 1);
  double dy = (Ymax - Ymin) / (M - 1);

  double *T = NULL;
  T = (double *)malloc(M * N * sizeof(double));

  for (int j = 0; j < M; j++) {
    for (int i = 0; i < N; i++) {
      x = Xmin + i * dx;
      y = Ymin + j * dy;

      // T[N * j + i] = (x - 2.) * (x ) * (y - 1.) * (y );
      T[N * j + i] = x * exp(y);
    }
  }

  VTK_out(N, M, &Xmin, &Xmax, &Ymin, &Ymax, T, 1);
  free(T);
  return(0);
}
*/

