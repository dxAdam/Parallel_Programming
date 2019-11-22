#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <math.h>
#include <mutex>


/* Compilation:
    gcc poissonPthreads.c vtk.c -o poissonPthreads -lpthreads
*/

/* Usage:
   ./poissonPthreads <M> <N> <t>

    Include integer values for M and N as command line arguments.

    These will be the dimensions of the 2D grid.

        M = Number of points along y-axis (rows)
        N = Number of points along x-axis (columns)
        t = Number of threads
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

        example: M x N = 5 X 4 

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

double dx, dy;
int M, N, t;

pthread_barrier_t barrier; 
pthread_barrierattr_t attr;

std::mutex mtx;

double * T         = NULL;
double * T_prev    = NULL;
double * T_source  = NULL;



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

double T_global_change = 1.1e-12;

void *jacobi(void *args){
    int iterations_count      = 0;
    int max_iterations        = 1e6;
    double target_convergence = 1e-12;
    double T_largest_change   = 1.1e-12; // must start > target_convergence
    double * TmpSwp           = NULL;  // used to swap T and T_prev pointers

    int m = 1;
    int n = 1;

    int my_M = M;
    int my_N = N;

    struct timespec begin, end;
    double elapsed;

    pthread_barrier_wait(&barrier);

    //THREAD
    int *id = (int*) args;

    if(*id == 0 && t == 2){
          my_N = N/2 + 1;
    }
    
    
    if(*id == 1){
        n = N/2;
    }


    //define constants so we don't need to calculate while iterating
    double    dx2 = dx*dx;
    double    dy2 = dy*dy;
    double      C = 0.5/(dx2+dy2);
    double dx2dy2 = dx2*dy2;
    if(*id==0)
        clock_gettime(CLOCK_MONOTONIC, &begin);
    while(iterations_count<max_iterations &&
		                  T_global_change > target_convergence)
    {

        //printf("Thread: %d!   T_largest_change: %f\n", *id, T_largest_change);

        // swap T and T_prev pointers if not first iteration
        if(iterations_count++ > 0 && *id==0)
        {
            TmpSwp = T_prev;
            T_prev = T;
            T = TmpSwp;
        }

        pthread_barrier_wait(&barrier);

        // reset largest change for this iteration
        T_largest_change = 0;
  
        for(int i=n; i<my_N-1; i++)
        {
            for(int j=m; j<my_M-1; j++)
            {
                // calculate new T(i,j) 
                T[i*my_M+j] =     C*((T_prev[my_M*i+j-1]            // below
                              +   T_prev[my_M*i+j+1])*dx2           // above
                              +  (T_prev[my_M*(i-1)+j]              // left
                              +   T_prev[my_M*(i+1)+j])*dy2         // right
                              -   T_source[my_M*i+j]*dx2dy2);   

	        	
                // calculate T_largest_change
                if(T[i*my_M+j]-T_prev[i*my_M+j] > T_largest_change)
                {
                    T_largest_change = T[i*my_M+j]-T_prev[i*my_M+j];
                    if(T_largest_change < 0)
                        T_largest_change*(-1);
                }

            }
        }

        if(T_largest_change < T_global_change){ 
            mtx.lock();
            T_global_change = T_largest_change;
            mtx.unlock();
        }
  


        pthread_barrier_wait(&barrier);

 
    }  
 
    if(*id==0){
        clock_gettime(CLOCK_MONOTONIC, &end);
        elapsed = end.tv_sec - begin.tv_sec;
        elapsed += (end.tv_nsec - begin.tv_nsec) /1000000000.0;
    }
    int i_max, j_max;
    double max_norm;

    max_norm = calculate_max_norm(M,N,T,T_source,&i_max,&j_max);

    if(*id == 0){
        printf("elapsed: %f seconds\n", elapsed);
        printf("iterations: %d\n", iterations_count);
        printf("max norm: %.12e at (%d,%d)\n", max_norm, i_max, j_max);
    }
}


int main(int argc, char *argv[2])
{
    M          = atoi(argv[1]); // number of points along y-axis (rows)
    N          = atoi(argv[2]); // number points along x-axis (cols)
    t          = atoi(argv[3]);

    M          = M + 2;         // this creates an extra layer around
    N          = N + 2;         //  our grid for boundary conditions

    // find dx and dy from boundary conditions and input
    dx         = (X_MAX - X_MIN) / (N-1);
    dy         = (Y_MAX - Y_MIN) / (M-1);

    // prepare arrays to hold calculated , exact, and previous array values

    T          = (double *)calloc(M*N, sizeof(double));
    T_prev     = (double *)calloc(M*N, sizeof(double));
    T_source   = (double *)calloc(M*N, sizeof(double));


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



    /* HANDLE THREADS */

    pthread_t thr[t];
    pthread_attr_t a;
    int id[t], i;

    pthread_barrier_init(&barrier, NULL, t);

    //create threads
    for(i=0; i<t; i++){
        id[i] = i;
        pthread_attr_init(&a);
        if(pthread_create(&thr[i], &a, jacobi, (void*)&id[i]) != 0)
            printf("Unable to create thread %d!\n", i);
    }
    
    //printf("Created %d threads ...\n", t);

    for(i=0; i<t; i++)
        pthread_join(thr[i], NULL);

    pthread_barrier_destroy(&barrier);









  //  printf("T:\n");
  //  print_table(M,N,T);

  //  printf("T_source:\n");
  //  print_table(M,N,T_source);
  



    // generate paraview .vtk file
    VTK_out(N, M, &X_MIN, &X_MAX, &Y_MIN, &Y_MAX, T, 0);
    
    free(T);
    free(T_source);
    free(T_prev);
}
