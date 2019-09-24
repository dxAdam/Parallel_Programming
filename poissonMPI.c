#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>

/*
    Compilation:
        mpicc poissonMPI.c -o poissonMPI -lm -O3
*/


/* 
    Program: poissonMPI

        Block paritioning Poisson solver that accepts number of processors (np) 
            and grid dimensions N X M as command line arguments. The number of points
            M and N must be a multiple of the number of processors in the respective direction.

/* 
    Usage:
            mpirun -np <p> poissonMPI <N> <M> <P>

            <p> - number of processors

            <M> - rows

            <N> - columns

            <P> - partition method:
            	    V - Vertical
            	    H - Horiztontal
            	    G - 2D Grid

    Example:
            mpirun -np 4 poissonMPI 40 80 G

        will divide a 40 x 80 matrix evenly among 4 processors in a 2D block pattern

        Note: 40 and 80 can be evenly divided between the processors in each direction. 
            So this is a valid configuration. Each processor will compute a 20x40 grid.


    Include integer values for np N and M as command line arguments.

    These will be the dimensions of the 2D grid.

        N = Number of points along x-axis
        M = Number of points along y-axis
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


       y
       ^
     1 |
       |
       |
       |
       |_ _ _ _ _ _ _ _ _ _> x
                          2

*/
//    //   //  // ////////////////////////////////////////////////////////////////////////////

//

// define MIN & MAX described above
double X_MIN = 0;
double X_MAX = 2;
double Y_MIN = 0;
double Y_MAX = 1;

double dx, dy;

char PRINT = 1; // PRINT == 0 only prints global norm and timing
            
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
    
    if(my_rank == 0 && PRINT){
        printf("%d x %d   %dp  2D Block Decomposition\n", M, N, np);
    }
    // used with MPI_Cart_create
    int dim[2], period[2], reorder;
    int coord[2], id;

    // find dx and dy from boundary conditions and input
    dx = (X_MAX - X_MIN) / (N+1);
    dy = (Y_MAX - Y_MIN) / (M+1);

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

    my_M_max = my_M_min + my_M - 1;
    my_N_max = my_N_min + my_N - 1;

}


/*
    Attempts to evenly divide grid horizontally among processors. Exits on failure
*/
void decompose_grid_horz(){

    int error = 0;

    if(my_rank == 0 && PRINT){
        printf("%d x %d  %dp  Horizontal decomposition\n", N, M, np);
    }

    // used with MPI_Cart_create
    int dim[2], period[2], reorder;
    int coord[2], id;

    // find dx and dy from boundary conditions and input
    dx = (X_MAX - X_MIN) / (N);
    dy = (Y_MAX - Y_MIN) / (M);

    my_N = N;
    my_N_min = 0;
    my_N_max = my_N_min + my_N - 1;

    my_M = M / np;
    my_M_min = my_rank*my_M;
    my_M_max = my_M_min + my_M - 1;
   
   
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

    if(my_rank == 0 && PRINT){
        printf("%d x %d  %dp  Vertical Decomposition\n", N, M, np);
    }

    // used with MPI_Cart_create
    int dim[2], period[2], reorder;
    int coord[2], id;

    // find dx and dy from boundary conditions and input
    dx = (X_MAX - X_MIN) / (N);
    dy = (Y_MAX - Y_MIN) / (M);

    my_N = N / np;
    my_N_min = my_rank*my_N;
    my_N_max = my_N_min + my_N-1;

    my_M = M;
    my_M_min = 0;
    my_M_max = my_M_min + my_M - 1;

    printf("my_rank:%d\nmy_M:%d  my_M_min:%d  my_M_max:%d\nmy_N:%d  my_N_min:%d  my_N_max:%d\n"
        ,my_rank, my_M, my_M_min, my_M_max, my_N, my_N_min, my_N_max);

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

    dim[0] = 1;
    dim[1] = np;
    period[0] = period[1] = reorder = 0;
    
    MPI_Cart_create(MPI_COMM_WORLD,2,dim,period,reorder,&com2d);
    MPI_Cart_shift(com2d, 1, 1, &left, &right);
    MPI_Cart_shift(com2d, 0, 1, &down, &up);
}


/*
    populates array with vertical boundary values
*/
void calculate_vert_boundaries(double *T, double *T_exact, int my_rank){
    
    int i,j;

    if(my_N_min == 0){  // along left bound
        for(j = 0; j<my_M+2; j++){
            T[j*(my_N + 2)] = T_exact[j*N+j*2];
        }
    }
    
    if(my_N_max == N-1) // along right bound
    {
        for(j = 0; j<my_M+2; j++){
            T[j*(my_N+2)+my_N+1] = T_exact[(j+my_M_min+1)*(N+2) - 1];
        }
    }
}


/*
    populates array with horizontal boundary values
*/
int calculate_horz_boundaries(double *T, double *T_exact, int my_rank){

    int i,j;

    if(my_M_min == 0){
        // top
        for(i = 0; i<my_N+2; i++){
            T[i] = T_exact[my_N_min + i];
        }
    }

    if(my_M_max == M-1){
        // bottom

        for(i = 0; i<my_N+2; i++){
            T[(my_M+1)*(my_N+2) + i] = T_exact[(M+1)*(N+2) + my_N_min + i];
        }
    }
}


/*
    Calculates and returns the max norm of the difference vectors from T and T_exact.
        Also finds the (x,y) location where the max norm occured.
*/
double calculate_max_norm(double *T, double *T_exact,int *i_max, int *j_max){
    double norm, max_norm = 0;
    int i,j;

    for(j=1;j<my_M+1;j++)
    {
        for(i=1;i<my_N+1;i++){
            //calculate the differnece vector and compare to current max
            norm = T[j*(my_N+2) + i] - T_exact[(j+my_M_min)*(N+2) + i + my_N_min];
            
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
    copied from provided vtk.c file. Used to generate .vtk file for Paraview visualizer.
*/
void VTK_out(const int N, const int M, const double *Xmin, const double *Xmax,
             const double *Ymin, const double *Ymax, const double *T,
             const int index) {
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
    Populate arrays with initial values
*/
void initialize_arrays(double **T, double **T_source, double **T_prev, double **T_exact){
    int i,j;
    for(i = 0; i<(my_N+2); i++){
        T[i] = &T[0][i*(my_M+2)];
        T_prev[i] = &T_prev[0][i*(my_M+2)];
    }

    // calculate T_exact using the source function for each entry
    for(j=0;j<M+2;j++){
        for(i=0;i<N+2;i++){
            T_exact[0][j*(N+2) + i] = source_function(X_MIN+i*dx,Y_MIN + j*dy);
        }
    }

    // calculate T_exact using the source function for each entry
    for(j=0;j<my_M+2;j++){
        for(i=0;i<my_N+2;i++){
            T_source[0][j*(my_N+2) + i] = source_function((i+my_N_min)*dx,(j+my_M_min)*dy);
        }
    }

}


/*
    Performs one Jacobi iteration through matrix   
*/
double jacobi(double *T, double *T_prev, double *T_source, int M, int N){

        double T_largest_change = 0;

        double C = 0.5/(dx*dx+dy*dy);
        double dx2 = dx*dx;
        double dy2 = dy*dy;

        // perform iteration
        for(int j = 1; j < M-1; j++){
            for(int i = 1; i < N-1; i++){
                // calculate new T(i,j) -- check report for derivation
                T[j*N + i] = C*((T[j*N + i-1]+T[j*N + i+1])*dy2 \
                     + (T[(j+1)*N + i]+T[(j-1)*N + i])*dx2 \
                        -  T_source[j*N + i]*dx2*dy2); 
                
                
                // calculate T_largest_change
                if(T[j*N+i]-T_prev[j*N+i] > T_largest_change)
                {
                    T_largest_change = T[j*N+i]-T_prev[j*N+i];
                    if(T_largest_change < 0)
                        T_largest_change*(-1);
                }
            }
        }


        for(int j = 1; j < M-1; j++){
            for(int i = 1; i < N-1; i++){
                //T_largest_change = fmax( fabs(T[j*N+i]-T_prev[j*N+i]), T_largest_change);
                T_prev[j*N + i] = T[j*N + i];
            }
        }


    return T_largest_change;
}



/*
    swaps rows with neighboring processes defined by MPI_Cart_create in main()  
*/
void swap_rows(double* T){

    MPI_Sendrecv(&T[(my_N+2)*(my_M)], 1, x_vector, up, 1, &T[0], 1, x_vector, down, 1, com2d, &status);
    MPI_Sendrecv(&T[my_N+2], 1, x_vector, down, 1, &T[(my_M+1)*(my_N+2)], 1, x_vector, up, 1, com2d, &status);
}

/*
    swaps columns with neighboring processes defined by MPI_Cart_create in main()  
*/
void swap_columns(double* T){

    MPI_Sendrecv(&T[my_N], 1, y_vector, right, 1, &T[0], 1, y_vector, left, 1, com2d, &status);
    MPI_Sendrecv(&T[1], 1, y_vector, left, 1, &T[my_N+1], 1, y_vector, right, 1, com2d, &status);
}


/*
    Fills an array with a hard-coded value below. Can be used to give matrix a starting
        value or for debugging.     
*/
void fill_array(double *T, int N, int M){

    int i,j;
    int x = 1;
    for(i=0; i<M; i++){
        for(j=0; j<N; j++){
            T[i*(N) + j] = 2*my_rank+2;
        }
        x++;
    }
}


/*
    prints a passed array T as a grid. Useful for troubleshooting.
*/
void print_tables(const int N, const int M, double *T)
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
    prints all arrays with barriers to prevent buffer collisions
*/
void print_all(double *T, double *T_exact){

    int i;
    for(i=0; i<np; i++){
        MPI_Barrier(com2d);
        if(my_rank == i){
          if(my_rank == 0){
              printf("T_exact:\n");
              print_tables(N+2, M+2, T_exact);
          }
            printf("\n process: %d   my_M: %d\n",i, my_M);
            print_tables(my_N+2, my_M+2, T);
        }
    }
}

//

//    //   //  // ////////////////////////////////////////////////////////////////////////////////////////////


int main (int argc, char* argv[]){

    double target_convergence = 10e-12;

    // infinity norm of processed matrix
    double my_max_norm = 0;
    double global_max_norm = 0;

    // change from one iteration to next
    double my_largest_change = 0; 
    double global_largest_change = 2*target_convergence;
    int iterations = 0;

    // used with MPI_Wtime()
    double start_time, end_time;

    char partition_method; // read from command line
    
    int iteration_limit = 10000000;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&np);  

    if(argc < 3){
        if(my_rank==0){
            printf("\n N M not detcted\n\n");
            printf("  usage:  mpirun -np <p> poissonMPI <N> <M>  <P> \n");
            printf("  <p> - number of processors\n  <N> - columns\n  <M> - rows\n"); 
            printf("  <P> - partition method:\n\tV - Vertical\n\tH - Horiztontal\n\tG - 2D Grid\n\n");
            printf("program terminating\n");
        }
        MPI_Finalize();
        exit(1);
    }
    else{
        N = atoi(argv[1]); // number of points along x-axis
        M = atoi(argv[2]); // number points along y-axis
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

    // declare type vectors for ghost rows and columns
    MPI_Type_vector(my_N+2, 1, 1, MPI_DOUBLE, &x_vector);
    MPI_Type_vector(my_M+2, 1, my_N+2, MPI_DOUBLE, &y_vector);
    MPI_Type_commit(&x_vector);
    MPI_Type_commit(&y_vector);

    // declare arrays then call initializing function
    double** T = (double **)malloc(sizeof(*T)*(my_N+2));
    T[0] = (double *)malloc(sizeof(double)*(my_N+2)*(my_M+2));

    double** T_source = (double **)malloc(sizeof(*T_source)*(my_N+2));
    T_source[0] = (double *)malloc(sizeof(double)*(my_N+2)*(my_M+2));

    double** T_prev = (double **)malloc(sizeof(*T_prev)*(my_N+2));
    T_prev[0] = (double *)malloc(sizeof(double)*(my_N+2)*(my_M+2));

    double** T_exact = (double **)malloc(sizeof(*T_exact)*(N+2));
        T_exact[0] = (double *)malloc(sizeof(double)*(N+2)*(M+2));

    double** TmpSwp = NULL;

    initialize_arrays(T, T_source, T_prev, T_exact);


    // this function can be used to populate the arrays with a starting value if desired
    //fill_array(T[0], my_N+2, my_M+2);

    if(up == MPI_PROC_NULL || down == MPI_PROC_NULL){
        calculate_horz_boundaries(T[0], T_exact[0], my_rank);
    }

    if(left == MPI_PROC_NULL || right == MPI_PROC_NULL){
        calculate_vert_boundaries(T[0], T_exact[0], my_rank);
    }


 
    start_time = MPI_Wtime();

    // begin Jacobi iterations
    //while(iterations < iteration_limit){
    while(global_largest_change > target_convergence && iterations < iteration_limit){

        swap_columns(T[0]);
        swap_rows(T[0]);

        my_largest_change = jacobi(T[0], T_prev[0], T_source[0], my_M+2, my_N+2);

        //// swap T and T_prev pointers if not first iteration
        //if(iterations++ > 0)
        //{
        //    TmpSwp = T_prev;
        //    T_prev = T;
        //    T = TmpSwp;
        //}

        MPI_Allreduce(&my_largest_change, &global_largest_change, 1, MPI_DOUBLE, MPI_MAX, com2d);
    }

    end_time = MPI_Wtime();
    

    //print_all(T[0], T_exact[0]);

    // generate .vtk files
    VTK_out(my_N+2, my_M+2, &X_MIN, &X_MAX, &Y_MIN, &Y_MAX, T[0], my_rank-1);

    int i,j; //used to retrieve max norm position
    my_max_norm = calculate_max_norm(T[0],T_exact[0],&i,&j);    
    MPI_Allreduce(&my_max_norm, &global_max_norm, 1, MPI_DOUBLE, MPI_MAX, com2d);

    // print max_norm and cleanup messages
    if(my_rank == 0){
        printf("max_norm: %.12e   time: %fs   iterations: %d\n",global_max_norm, end_time - start_time, iterations);
    }

    free(T);
    free(T_source);
    free(T_prev);
    free(T_exact);

    MPI_Finalize();
    return(0);
} 
