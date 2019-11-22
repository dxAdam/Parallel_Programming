CC=g++
CFLAGS= -lpthread
poissonSerial: poissonPthreads.cpp vtk.c
	$(CC) -o poissonPthreads poissonPthreads.cpp vtk.c $(CFLAGS)

clean:
	rm -f *.o *.vtk poissonPthreads
