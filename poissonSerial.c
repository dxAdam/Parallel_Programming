#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "openacc.h"

/* Compilation:
    Serial: gcc poissonACC.c -o poissonACC -lm
    Parallel: pgcc -acc -Minfo -ta=tesla:cuda9.1 -fast -O3 poissonACC.c -o poissonACC
*/

/* Usage:
   ./poissonACC <M> <N>

    Include integer values for M and N as command line arguments.

    These will be the dimensions of the 2D grid.

        M = Number of points along y-axis (rows)
        N = Number of points along x-axis (columns)
*/

/*
    Source term:

       S(x,y) = x*exp(y)

       note: x*exp(y) is an exact solution of the Laplace equation.

    Boundary conditions:

       0 <= x <= 2
       0 <= y <= 1

       T(0,y) = 0             (left)
       T(2,y) = 2*exp(y)      (right)
       T(x,0) = x             (bottom)
       T(x,1) = x*exp(1)      (top)
                          

*/
// / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / /

// define MIN & MAX described above
double X_MIN = 0;
double X_MAX = 2;
double Y_MIN = 0;
double Y_MAX = 1;


/*
    this function calculates and returns the value of the
      source function S(x,y) = x*exp(y) at (x,y)
*/
double source_function(double x, double y)
{
    return x*exp(y);
}


/*
    prints a passed array T as a grid. Useful for troubleshooting.
*/
void print_table(const int M, const int N, double *T)
{

    int i,j;
    double tmp;

    for(j=0;j<=M+1;j++)
    {
        for(i=0;i<=N+1;i++)
        {
            tmp = T[(M+1)*(i+1) + i - j];
            printf("%.6f ",tmp);
        }
        printf("\n");
    }
    printf("\n");
}


/*
    calculates and populates boundaries for passed array T based on function information above.
        The dx and dy scaling factors are applied here.
*/
void calculate_boundaries(const int M, const int N, double *T, double dx, double dy)
{

    #pragma acc data copyin(T[0:M*N]) copyout(T[0:N*M])

    for(int i=0; i <= N+1; i++)
    {
        // bottom
        T[i*(M+2)] = source_function(X_MIN+i*dx,Y_MIN);
        // top
        T[i*(M+2) + M+1] = source_function(X_MIN+i*dx,Y_MAX);
    }

    for(int j=0; j<=M+1; j++)
    {
        //left
        T[j] = source_function(X_MIN,Y_MIN+j*dy);
        //right
        T[(M+2)*(N+1) + j] = source_function(X_MAX,Y_MIN+j*dy);
    }
}


/*
    copied from provided vtk.c file. Used to generate .vtk file for Paraview visualizer.
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


/*
    Calculates and returns the max norm of the difference vectors from T and T_exact.
        Also finds the (x,y) location where the max norm occured.
*/
double calculate_max_norm(const int M,const int N, double *T, 
                                double *T_exact,int *i_max, int *j_max)
{
    double norm, max_norm = 0;

    for(int i=1;i<=N;i++)
    {
        for(int j=1;j<=M;j++)
        {
            //calculate the differnece vector and compare to current max
            norm = T[i*(M+2)+j] - T_exact[i*(M+2)+j];
            
            if(norm < 0)  // get absolute value if negative
                norm = norm*(-1);

            if(norm > max_norm)
            {
                max_norm = norm;
                *i_max = i;
                *j_max = j;
            }
        }
    }
    return max_norm;
}




int main(int argc, char *argv[2])
{
    int M                     = atoi(argv[1]); // number of points along y-axis (rows)
    int N                     = atoi(argv[2]); // number points along x-axis (cols)
    
    int iterations_count      = 0;
    double target_convergence = 10e-12;
    double T_largest_change   = target_convergence + 1; // must start greater than target_convergence

    // find dx and dy from boundary conditions and input
    double dx                 = (X_MAX - X_MIN) / (N+1);
    double dy                 = (Y_MAX - Y_MIN) / (M+1);

    // prepare arrays to hold calculated , exact, and previous array values
    double * T                = NULL;
    double * T_prev           = NULL;
    double * TmpSwp           = NULL;  // used to swap T and T_prev pointers
    double *restrict T_exact  = NULL;
    T                         = (double *)calloc((M + 2) * (N + 2), sizeof(double));
    T_prev                    = (double *)calloc((M + 2) * (N + 2), sizeof(double));
    T_exact                   = (double *)calloc((M + 2) * (N + 2), sizeof(double));



#pragma acc data copyin(T_exact[0:M*N]) copyout(T_exact[0:N*M])
{
    // calculate T_exact using the source function for each entry
    #pragma acc parallel loop gang

    for(int i=0;i<=N+1;i++)
    {
        //#pragma acc loop vector
        for(int j=0;j<=M+1;j++)
            T_exact[i*(M+2)+j] = source_function(X_MIN+i*dx,Y_MIN+j*dy);
    }
}

    // calculate the boundries defined by the specific problem for T and T_prev
    calculate_boundaries(M,N,T,dx,dy);
    calculate_boundaries(M,N,T_prev,dx,dy);

    clock_t start = clock();


#pragma acc data copyin(T[0:M*N], T_prev[0:M*N], T_exact[0:M*N]) copyout(T[0:N*M])
{
    
    // Begin Jacobi iterations
    while(T_largest_change > target_convergence)
    //while(iterations_count<100000 && T_largest_change > target_convergence)
    {   

        //define constants so we don't need to calculate while iterating
        double    dx2 = dx*dx;
        double    dy2 = dy*dy;
        double      C = 0.5/(dx2+dy2);
        double dx2dy2 = dx2*dy2;
        int        M2 = M+2;

        // swap T and T_prev pointers if not first iteration
        if(iterations_count++ > 0)
        {
            TmpSwp = T_prev;
            T_prev = T;
            T = TmpSwp;
        }

        // reset largest change for this iteration
        T_largest_change = 0;
          
        #pragma acc parallel loop gang
        for(int i=1; i<=N; i++)
        {
            #pragma acc loop vector
            for(int j=1; j<=M; j++)
            {
                // calculate new T(i,j) 
                T[i*(M2)+j] =   C*((T_prev[i*(M2)+j-1]              // below
                                +   T_prev[i*(M2)+j+1])*dx2         // above
                                +  (T_prev[(i-1)*(M2)+j]            // left
                                +   T_prev[(i+1)*(M2)+j])*dy2       // right
                                -   T_exact[i*(M2)+j]*dx2dy2);   

                
                // calculate T_largest_change
                if(T[i*(M2)+j]-T_prev[i*(M2)+j] > T_largest_change)
                {
                    T_largest_change = T[i*(M2)+j]-T_prev[i*(M2)+j];
                    if(T_largest_change < 0)
                        T_largest_change*(-1);
                }
            }
        }
    }
}

    clock_t stop = clock();

    int i_max;
    int j_max;
    double max_norm;

    // determine the max_norm of the difference vectors between T and T_exact and print
    max_norm = calculate_max_norm(M,N,T,T_exact,&i_max,&j_max);

    printf("elapsed: %f seconds\n", (double)(stop - start)/1000000);
    printf("iterations: %d\n", iterations_count);
    printf("max norm: %.12e at (%d,%d)\n", max_norm, i_max, j_max);

    /*
    printf("T:\n");
    print_table(M,N,T);

    printf("T_exact:\n");
    print_table(M,N,T_exact);
    */

    // generate paraview .vtk file
    VTK_out(M+2, N+2, &X_MIN, &X_MAX, &Y_MIN, &Y_MAX, T, 0);
    
    free(T);
    free(T_exact);
    free(T_prev);
}
