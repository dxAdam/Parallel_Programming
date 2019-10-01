CC=mpicc
CFLAGS=-std=c99 -lm
poissonMPI: poissonMPI.c vtk.c
	$(CC) -o poissonMPI poissonMPI.c vtk.c $(CFLAGS)

clean:
	rm -f *.o *.vtk poissonMPI
