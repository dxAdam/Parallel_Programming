#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>

/*
    Compilation:
        mpicc -std=c99 poissonMPI.c vtk.c -o poissonMPI -lm -O0
*/


/* 
    Program: poissonMPI

        Block paritioning Poisson solver that accepts number of processors (np) 
        and grid dimensions M X N as command line arguments. The number of 
        points M and N must be a multiple of the number of processors in the
        respective direction.
*/


/*
    Usage:

        mpirun -np <p> poissonMPI <N> <M> <P> <I>

            <p> - number of processors

            <M> - rows

            <N> - columns

            <P> - partition method:  (Optional - defaults to Vertical)
            	    V - Vertical
            	    H - Horiztontal
            	    G - 2D Grid

            <I> - max iterations     (Optional - defaults to 1e6)

    Example:
        
        mpirun -np 4 poissonMPI 40 80 G

        will divide a 40 x 80 matrix evenly among 4 processors in a 2D block
        pattern (G)

        Note: 40 and 80 can be evenly divided between the processors in each
        direction. 
            
        So this is a valid configuration. Each processor will compute a
        20x40 grid.

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
        T(x,1) = x*exp(1)        (top)

    
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
//    //   //  // ///////////////////////////////////////////////////////////////////////


// define MIN & MAX described above
double GLOBAL_X_MIN = 0;
double GLOBAL_X_MAX = 2;
double GLOBAL_Y_MIN = 0;
double GLOBAL_Y_MAX = 1;

double dx, dy;

char DEBUG = 1; // DEBUG == 1 prints extra information useful for debugging
            
MPI_Status status;
MPI_Datatype x_vector, y_vector;
MPI_Comm com2d;

int ny, nx; // number of processors in y (M) and x (N) direction

int my_rank,np, N, M, my_N, my_M;
int my_N_min, my_N_max, my_M_min, my_M_max, nx, ny;    

int up, down, left, right; // rank of process in respective direction

/*
    this function calculates and returns the value of the
      source function S(x,y) = x*exp(y) at (x,y)
*/
double source_function(double x, double y){
    return x*exp(y);
}


/*
    Attempts to evenly divide grid in a 2D block pattern. Exits on failure.
*/
void decompose_grid_2D_block(){

    int error = 0;
    
    if(my_rank == 0){
        printf("%d x %d   %dp  2D Block Decomposition\n", M, N, np);
    }
    // used with MPI_Cart_create
    int dim[2], period[2], reorder;
    int coord[2], id;

    // find dx and dy from boundary conditions and input
    dx = (GLOBAL_X_MAX - GLOBAL_X_MIN) / (N+1);
    dy = (GLOBAL_Y_MAX - GLOBAL_Y_MIN) / (M+1);

    if(np == 1)
    {
        nx = ny = 1;
    }
    else{

        if(M == N){
            nx = ny = sqrt(np);
            if(nx*ny != np){
                 error = 1;
            }  
        }
        else if(M % N == 0){ // M is multiple of N
            double m = M;
            double n = N;
            double p = np;

            nx = M / N;
            ny = p / nx;
        }
        else if(N % M == 0){ // N is multiple of M
            double m = M;
            double n = N;
            double p = np;

            ny = N / M;
            nx = p / ny;
        }
        else{
            // now we need to check that my_M*np==M to make sure   *NOTE: did not finish this part
            //   we're not losing any of the original matrix
            if(my_M*np != M){
                error = 1;
                MPI_Bcast(&error, 1, MPI_INT, 0, MPI_COMM_WORLD);
                if(error != 0) {
                    if(my_rank == 0){
                        printf("M=%d is not evening divisible by number of processors=%d\n\nprogram will now terminate\n", M, np);
                    }
                MPI_Finalize();
                exit(error);
                }
            }
        }
    }
    
    // define values used with MPI_CART functions below
    dim[0] = ny;
    dim[1] = nx;
    period[0] = period[1] = reorder = 0;

    // create coordinate system and get coords
    MPI_Cart_create(MPI_COMM_WORLD,2,dim,period,reorder,&com2d);
    MPI_Cart_coords(com2d, my_rank, 2, coord);

    // find neighbors
    MPI_Cart_shift(com2d, 1, 1, &left, &right);
    MPI_Cart_shift(com2d, 0, 1, &down, &up);
    

    my_M = M / ny;
    my_N = N / nx;

    my_N_min = my_N*coord[1];
    my_M_min = my_M*coord[0];

    my_M = my_M + 2;
    my_N = my_N + 2;

    my_M_max = my_M_min + my_M - 1;
    my_N_max = my_N_min + my_N - 1;

}


/*
    Attempts to evenly divide grid horizontally among processors. Exits on failure
*/
void decompose_grid_horz(){

        int error = 0;

    if(my_rank == 0){
        printf("%d x %d  %dp  Horizontal decomposition\n", N, M, np);
    }

    // used with MPI_Cart_create
    int dim[2], period[2], reorder;
    int coord[2], id;

    // find dx and dy from boundary conditions and input
    dx = (GLOBAL_X_MAX - GLOBAL_X_MIN) / (N+1);
    dy = (GLOBAL_Y_MAX - GLOBAL_Y_MIN) / (M+1);


    my_M = M / np;
   
    // now we need to check that my_M*np==M to make sure
    //   we're not losing any of the original matrix
    if(my_M*np != M){
        error = 1;
        MPI_Bcast(&error, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if(error != 0) {
            if(my_rank == 0){
                printf("M=%d is not evening divisible by number of processors=%d\n\nprogram will now terminate\n", M, np);
            }
            MPI_Finalize();
            exit(error);
        }
    }
    
    my_M_min = my_rank*my_M;

    my_M = my_M + 2;
    my_M_max = my_M_min + my_M - 1;

    my_N = N;
    my_N_min = 0;
    my_N = my_N + 2;
    my_N_max = my_N_min + my_N - 1;


    dim[0] = np;
    dim[1] = 1;
    period[0] = period[1] = reorder = 0;

    MPI_Cart_create(MPI_COMM_WORLD,2,dim,period,reorder,&com2d);
    MPI_Cart_shift(com2d, 1, 1, &left, &right);
    MPI_Cart_shift(com2d, 0, 1, &down, &up);
    
}


/*
    Attempts to evenly divide grid vertically among processors. Exits on failure.
*/
void decompose_grid_vert(){
    int error = 0;

    if(my_rank == 0){
        printf("%d x %d  %dp  Vertical Decomposition\n", N, M, np);
    }

    // used with MPI_Cart_create
    int dims, dim[2], period[2], reorder;
    int coord[2], id;

    // find dx and dy from boundary conditions and input
    dx = (GLOBAL_X_MAX - GLOBAL_X_MIN) / (N+1);
    dy = (GLOBAL_Y_MAX - GLOBAL_Y_MIN) / (M+1);

    my_N = N / np;

    // now we need to check that my_N*np == N to make sure
    //   we're not losing any of the original matrix
    if(my_N*np != N){
        error = 1;
        MPI_Bcast(&error, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if(error != 0) {
            if(my_rank == 0){
                printf("N=%d is not evening divisible by number of processors=%d\n\nprogram will now terminate\n", N, np);
            }
            MPI_Finalize();
            exit(error);
        }
    }

    my_N_min = my_rank*my_N;
    my_N = my_N + 2;
    my_N_max = my_N_min + my_N - 1;
    

    my_M = M;
    my_M_min = 0;
    my_M = my_M + 2;
    my_M_max = my_M_min + my_M - 1;



    dim[0] = 1;    // rows
    dim[1] = np;   // cols
    period[0] = period[1] = reorder = 0;
    
    MPI_Cart_create(MPI_COMM_WORLD,2,dim,period,reorder,&com2d);
    MPI_Cart_shift(com2d, 1, 1, &left, &right);
    MPI_Cart_shift(com2d, 0, 1, &down, &up);
    
    
    
}


/*
    populates array with vertical boundary values
*/
void calculate_vert_boundaries(double *T, double *T_source){
    
    int i,j;
    int x = 1;
    
    // left
    if(my_N_min == 0){
        for(i = 0; i < my_M; i++){
            T[i*my_N] = T_source[i*my_N];
        }
    }
    
    // right
    if(my_N_max == N+1)
    {
        for(i = 1; i < my_M+1; i++){
            T[my_N*i-1] = T_source[my_N*i-1];
        }
    }
}


/*
    populates array with horizontal boundary values
*/
int calculate_horz_boundaries(double *T, double *T_source){

    int i,j;
    
    // top
    if(my_M_min == 0){
        for(i = 0; i<my_N; i++){
            T[i] = T_source[i];
        }
    }

    // bottom
    if(my_M_max == M+1){
        for(i = 1; i<my_N; i++){
            T[my_N*(my_M-1) + i] = T_source[my_N*(my_M-1)+i];
        }
    }
}


/*
    Fills an array with a hard-coded value below. Can be used to give matrix a starting
        value or for debugging.     
*/
void fill_array(double *T)
{
    int i,j;
    int x = 1 + 100*my_rank;
    for(j=1; j<my_M-1; j++){
        for(i=1; i<my_N-1; i++){
            T[my_N*j + i] = x++;//i*my_M + j;
        }
    }
}


/*
    Calculates and returns the max norm of the difference vectors from T and T_source.
        Also finds the (x,y) location where the max norm occured.
*/
double calculate_max_norm(double *T, double *T_source,int *i_max, int *j_max){
    double norm, max_norm = 0;
    int i,j;

    for(j=1;j<my_M-1; j++)
    {
        for(i=1;i<my_N-1;i++){

            //calculate the differnece vector and compare to current max
            norm = T[my_N*j + i]  -  T_source[my_N*j + i];

            if(norm < 0)  // get absolute value if negative
                norm = norm*(-1);

            if(norm > max_norm){
                max_norm = norm;
                *i_max = i;
                *j_max = j;
            }
        }
    }

    return max_norm;
}


/*
     trim the edges off a matrix and return a new one in its place. We need this
     because the T array is (my_M+2) x (my_N+2) due to ghost cells, but now it
     will be easier to have T and T_source the same size.
*/
double * trim_matrix_edges(double *T)
{
   
    double *Tmp = (double *)calloc(sizeof(double)*(my_M-2)*(my_N-2), sizeof(double));

    int i, j;

    int x=1;

    for(j=0;j<my_M-2; j++)
    {
        for(i=0;i<my_N-2;i++)
        {
            Tmp[(my_N-2)*j+i] = T[(my_N)*(j+1) + 1+i];
        }
    }
    free(T);
    return Tmp;
}


/*
    located in vtk.c
*/
extern void VTK_out(const int N, const int M, const double *Xmin, const double *Xmax,
             const double *Ymin, const double *Ymax, const double *T,
             const int index);


/*
    Populate arrays with initial values
*/
void initialize_arrays(double *T, double *T_prev, double *T_source, double X_MIN, double Y_MIN)
{    
    // calculate T_source using the source function for each entry
    int i,j;
    for(j=0;j<my_M;j++){
        for(i=0;i<my_N;i++){
            T_source[my_N*j + i] = source_function(X_MIN + i*dx,Y_MIN + j*dy);
        }
    }
}



/*
    swaps rows with neighboring processes defined by MPI_Cart_create in main()  
*/
void swap_rows(double* T)
{      

    MPI_Sendrecv(&T[my_N], 1, x_vector, down, 1, &T[(my_M-1)*my_N], 1, x_vector, up, 1, com2d, &status);                                                 //dest                        source
    MPI_Sendrecv( &T[(my_M-2)*my_N], 1, x_vector, up, 1,&T[0], 1, x_vector, down, 1, com2d, &status);
}



/*
    swaps columns with neighboring processes defined by MPI_Cart_create in main()  
*/
void swap_columns(double* T)
{
    MPI_Sendrecv(&T[1], 1, y_vector, left, 1, &T[my_N-1], 1, y_vector, right, 1, com2d, &status);
    MPI_Sendrecv(&T[my_N-2], 1, y_vector, right, 1, &T[0], 1, y_vector, left, 1, com2d, &status);
}



/*
    prints a passed array T as a grid. Useful for troubleshooting.
*/
void print_tables(double *T)
{
    int i,j;
    double tmp;

    for(j=my_M-1;j>=0;j--)
    {
        for(i=0;i<my_N;i++)
        {
            tmp = T[my_N*j + i];
            printf("%.6f ", tmp);
        }
        printf("\n");
    }
    printf("\n");
}


/*
    prints all arrays with barriers to prevent buffer collisions
*/
void print_all(double *T, double *T_source)
{
    int i;
    for(i=0; i<np; i++){
        MPI_Barrier(com2d);
        if(my_rank == i){
            printf("rank: %d my_M: %d my_N: %d T:\n", my_rank, my_M, my_N);
            print_tables(T);
            printf("rank: %d T_source:\n", my_rank);
            print_tables(T_source);
            fflush(stdout);
        }
    }
}


/*
    Performs one Jacobi iteration through matrix   
*/
double jacobi(double *T, double *T_prev, double *T_source)
{
        double T_largest_change = 0;

        // save time by calculating these values here
        double C = 0.5/(dx*dx+dy*dy);
        double dx2 = dx*dx;
        double dy2 = dy*dy;
        double dx2dy2 = dx2*dy2;

        for(int i = 1; i<my_N-1; i++)
        {
            for(int j=1; j<my_M-1; j++)
            {    
                // calculate new T(i,j)
                T[my_N*j + i] =  C*((T_prev[my_N*j +i-1]               // left
                                +    T_prev[my_N*j+i+1])*dy2           // right
                                +   (T_prev[my_N*(j-1)+i]              // below
                                +    T_prev[my_N*(j+1)+i])*dx2         // above
                                -    T_source[my_N*j + i]*dx2dy2);   


                // calculate T_largest_change
                if(T[my_N*j + i]-T_prev[my_N*j + i] > T_largest_change)
                {
                    T_largest_change = T[my_N*j + i]-T_prev[my_N*j + i];
                    if(T_largest_change < 0)
                        T_largest_change*(-1);
                }
            }
        }

    return T_largest_change;
}



//
//    //   //  // ////////////////////////////////////////////////////////////////////////////////////////////


int main (int argc, char* argv[]){

    int iteration_limit          = 1e6;
    int iterations               = 0;
    double target_convergence    = 10e-12;

    // infinity norm of processed matrix
    double my_max_norm           = 0;
    double global_max_norm       = 0;

    // change from one iteration to next
    double my_largest_change     = 0; 
    double global_largest_change = target_convergence + 1; //needs to be > than global to iterate

    // used with MPI_Wtime()
    double start_time, end_time;

    char partition_method; // read from command line


    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&np);  

    if(argc < 3){
        if(my_rank==0){
            printf("\n M N not detcted\n\n");
            printf("  usage:  mpirun -np <p> poissonMPI <M> <N>  <P> \n");
            printf("  <p> - number of processors\n  <N> - columns\n  <M> - rows\n"); 
            printf("  <P> - partition method:\n\tV - Vertical\n\tH - Horiztontal\n\tG - 2D Grid\n\n");
            printf("program terminating\n");
        }
        MPI_Finalize();
        exit(1);
    }
    else{
        M = atoi(argv[1]); // number of points along x-axis
        N = atoi(argv[2]); // number points along y-axis
        if(argc < 4){
            if(my_rank == 0)
                printf("Partition method not detected: Defaulting to Vertical Decomposition\n");
            partition_method = 'V';
        }
        else
            partition_method = argv[3][0];
        if(argc > 4)
            iteration_limit = atoi(argv[4]);
    }

    // divide grid among processors
    if(partition_method == 'V')
        decompose_grid_vert();
    else if(partition_method == 'H')
        decompose_grid_horz();
    else if(partition_method == 'G')
        decompose_grid_2D_block();
    else{
        if(my_rank == 0)
            printf("Partition method not detected: Defaulting to Vertical Decomposition\n");
        partition_method = 'V';
    }

    double X_MIN = my_N_min*dx;
    double X_MAX = my_N_max*dx;
    double Y_MIN = my_M_min*dy;
    double Y_MAX = my_M_max*dy;

    
    //printf("my_rank: %d\nmy_N_min: %d  my_N_max: %d   my_M_min: %d   my_M_max: %d\n", my_rank, my_N_min, my_N_max, my_M_min, my_M_max);
    //printf("X_MIN: %f   X_MAX: %f   Y_MIN: %f   Y_MAX: %f  dx: %f   dy: %f\n\n", X_MIN, X_MAX, Y_MIN, Y_MAX, dx,dy);

    //printf("\nmy_rank:%d\nmy_M:%d  my_M_min:%d  my_M_max:%d\nmy_N:%d  my_N_min:%d  my_N_max:%d\n\n"
    //    ,my_rank, my_M, my_M_min, my_M_max, my_N, my_N_min, my_N_max);

    // declare type vectors for ghost rows and columns
    MPI_Type_vector(my_N, my_N, 1, MPI_DOUBLE, &x_vector); //row
    MPI_Type_vector(my_M, 1, my_N, MPI_DOUBLE, &y_vector); //col
    MPI_Type_commit(&x_vector);
    MPI_Type_commit(&y_vector);

    
    double *T_prev        = (double *)calloc(sizeof(double)*my_M*my_N, sizeof(double));
    
    double *T             = (double *)calloc(sizeof(double)*my_M*my_N, sizeof(double));
    
    double *T_source      = (double *)calloc(sizeof(double)*my_M*my_N, sizeof(double));
    
    double *TmpSwp        = NULL;


    initialize_arrays(T, T_prev, T_source, X_MIN, Y_MIN);

    // this function can be used to populate the arrays with a starting value if desired

   //fill_array(T);

    calculate_horz_boundaries(T, T_source);
    calculate_horz_boundaries(T_prev, T_source);

    calculate_vert_boundaries(T, T_source);
    calculate_vert_boundaries(T_prev, T_source);


    start_time = MPI_Wtime();
    

    // begin Jacobi iterations
    while(iterations < iteration_limit && global_largest_change > target_convergence )
    //while(iterations < 1)
    {
        swap_columns(T);
        swap_rows(T);

        // swap T and T_prev pointers if not first iteration
        if(iterations++ > 0)
        {
            TmpSwp = T_prev;
            T_prev = T;
            T = TmpSwp;
        }

        my_largest_change = jacobi(T, T_prev, T_source);

        if(np == 1)
            global_largest_change = my_largest_change;
        else
            MPI_Allreduce(&my_largest_change, &global_largest_change, 1, MPI_DOUBLE, MPI_MAX, com2d);
    }
    

    end_time = MPI_Wtime();
    

    //print_all(T, T_source);

   
    // calculate max norm
    int i,j; //used to retrieve max norm position 
    my_max_norm = calculate_max_norm(T,T_source, &i,&j);    
    
    if(np == 1)
        global_max_norm = my_max_norm;
    else
        MPI_Allreduce(&my_max_norm, &global_max_norm, 1, MPI_DOUBLE, MPI_MAX, com2d);

    // print max_norm and cleanup messages
    if(my_rank == 0){
        printf("\nmax_norm: %.12e\ntime: %fs\niterations: %d\n",global_max_norm, end_time - start_time, iterations);
    }

    VTK_out(my_N, my_M, &X_MIN, &X_MAX, &Y_MIN, &Y_MAX, T, my_rank);

    free(T);
    free(T_prev);
    free(T_source);

    MPI_Finalize();
    return(0);
} 
