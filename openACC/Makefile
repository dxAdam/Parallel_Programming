CC=pgcc
CFLAGS=-acc -lm -Minfo

all: poissonACC_cpu poissonACC_CUDA

poissonACC_cpu: poissonACC.c vtk.c
	$(CC) -o poissonACC_cpu poissonACC.c vtk.c $(CFLAGS) -ta=multicore

poissonACC_CUDA: poissonACC.c vtk.c
	$(CC) -o poissonACC_CUDA poissonACC.c vtk.c $(CFLAGS) -ta=tesla

clean:
	rm -f *.o *.vtk poissonACC_cpu poissonACC_CUDA
