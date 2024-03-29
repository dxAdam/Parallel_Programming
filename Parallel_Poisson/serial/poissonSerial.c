#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Compilation:
    gcc poissonSerial.c vtk.c -o poissonSerial -lm
*/

/* Usage:
   ./poissonSerial <M> <N>

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
    
    
    Grid Layout:

        example: M x N = 6 X 4 

    (0,0)____________ i=N__x    * When we print, we mirror the x-axis so the
        |[ 0][ 1][ 2][ 3]         plot appears with 23 on the top right corner
        |[ 4][ 5][ 6][ 7]
        |[ 8][ 9][10][11]
        |[12][13][14][15]
        |[16][17][18][19]
    j=M |[20][21][22][23]
        y                        

*/
// / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / /

// define MIN & MAX described above
double X_MIN = 0;
double X_MAX = 2;
double Y_MIN = 0;
double Y_MAX = 1;

/*
    located in vtk.c
*/
extern void VTK_out(const int N, const int M, const double *Xmin, const double *Xmax,
             const double *Ymin, const double *Ymax, const double *T,
             const int index);



/*
    this function calculates and returns the value of the
      source function S(x,y) = x*exp(y) at (x,y)
*/
double source_function(double x, double y)
{
    return x*exp(y);
}



/*
    prints a passed array T as a grid. Useful for troubleshooting. Mirrors x-axis to appear upright
*/
void print_table(const int M, const int N, double *T)
{
    int i,j;
    double tmp;

    for(j=0;j<M;j++)
    {
        for(i=0;i<N;i++)
        {
            tmp = T[(M-1)*(i+1)+i-j];
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
    for(int i=0; i < N; i++)
    {
        // bottom
        T[M*i] = source_function(X_MIN+i*dx,Y_MIN);
        // top
        T[M*i + M - 1] = source_function(X_MIN+i*dx,Y_MAX);
    }

    for(int j=0; j < M; j++)
    {
        //left
        T[j] = source_function(X_MIN,Y_MIN+j*dy);
        //right
        T[M*(N-1) + j] = source_function(X_MAX,Y_MIN+j*dy);
    }
}



/*
    Calculates and returns the max norm of the difference vectors from T and T_source.
        Also finds the (x,y) location where the max norm occured.
*/
double calculate_max_norm(const int M,const int N, double *T, 
                                double *T_source,int *i_max, int *j_max)
{
    double norm, max_norm = 0;

    for(int i=1;i<N;i++)
    {
        for(int j=1;j<M;j++)
        {
            //calculate the differnece vector and compare to current max
            norm = T[i*M+j] - T_source[i*M+j];
            
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
    
    M                         = M + 2;         // this creates an extra layer around
    N                         = N + 2;         //  our grid for boundary conditions

    int iterations_count      = 0;
    int max_iterations        = 1e6;
    double target_convergence = 1e-12;
    double T_largest_change   = target_convergence + 1; // must start > target_convergence

    // find dx and dy from boundary conditions and input
    double dx                 = (X_MAX - X_MIN) / (N-1);
    double dy                 = (Y_MAX - Y_MIN) / (M-1);

    // prepare arrays to hold calculated , exact, and previous array values
    double * T                = NULL;
    double * T_prev           = NULL;
    double * TmpSwp           = NULL;  // used to swap T and T_prev pointers
    double *restrict T_source = NULL;
    T                         = (double *)calloc(M*N, sizeof(double));
    T_prev                    = (double *)calloc(M*N, sizeof(double));
    T_source                  = (double *)calloc(M*N, sizeof(double));



    // Initialize matrices

    // calculate T_source using the source function for each entry
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<M;j++)
            T_source[i*M+j] = source_function(X_MIN+i*dx,Y_MIN+j*dy);
    }

    // calculate the boundries defined by the specific problem for T and T_prev
    calculate_boundaries(M,N,T,dx,dy);
    calculate_boundaries(M,N,T_prev,dx,dy);



    // Perform Jacobi iterations
    
    clock_t start = clock();
    while(iterations_count<max_iterations &&
		                  T_largest_change > target_convergence)
    {   

        //define constants so we don't need to calculate while iterating
        double    dx2 = dx*dx;
        double    dy2 = dy*dy;
        double      C = 0.5/(dx2+dy2);
        double dx2dy2 = dx2*dy2;

        // swap T and T_prev pointers if not first iteration
        if(iterations_count++ > 0)
        {
            TmpSwp = T_prev;
            T_prev = T;
            T = TmpSwp;
        }

        // reset largest change for this iteration
        T_largest_change = 0;
          
        for(int i=1; i<N-1; i++)
        {
            for(int j=1; j<M-1; j++)
            {
                // calculate new T(i,j) 
                T[i*M+j] =    C*((T_prev[M*i+j-1]                // below
                              +   T_prev[M*i+j+1])*dx2           // above
                              +  (T_prev[M*(i-1)+j]              // left
                              +   T_prev[M*(i+1)+j])*dy2         // right
                              -   T_source[M*i+j]*dx2dy2);   

	        	
                // calculate T_largest_change
                if(T[i*M+j]-T_prev[i*M+j] > T_largest_change)
                {
                    T_largest_change = T[i*M+j]-T_prev[i*M+j];
                    if(T_largest_change < 0)
                        T_largest_change*(-1);
                }
            }
        }
    }
    clock_t stop = clock();



    // determine the max_norm of the difference vectors between T and T_source

    int i_max, j_max;
    double max_norm;

    max_norm = calculate_max_norm(M,N,T,T_source,&i_max,&j_max);

    printf("elapsed: %f seconds\n", (double)(stop - start)/1000000);
    printf("iterations: %d\n", iterations_count);
    printf("max norm: %.12e at (%d,%d)\n", max_norm, i_max, j_max);

/*
    printf("T:\n");
    print_table(M,N,T);

    printf("T_source:\n");
    print_table(M,N,T_source);
*/  



    // generate paraview .vtk file
    VTK_out(N, M, &X_MIN, &X_MAX, &Y_MIN, &Y_MAX, T, 0);
    
    free(T);
    free(T_source);
    free(T_prev);
}
