CC=gcc
CFLAGS=-std=c99 -lm -fopenmp
poissonOMP: poissonOMP.c vtk.c
	$(CC) -o poissonOMP poissonOMP.c vtk.c $(CFLAGS)

clean:
	rm -f *.o *.vtk poissonOMP
