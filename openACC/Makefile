CC=pgcc
CFLAGS=-acc -Minfo

all: poissonACC_cpu poissonACC_CUDA

poissonACC_cpu: poissonACC.c vtk.c
	$(CC) -o poissonACC_cpu poissonACC.c vtk.c $(CFLAGS) -ta=multicore -lm

poissonACC_CUDA: poissonACC.c vtk.c
	$(CC) -o poissonACC_CUDA poissonACC.c vtk.c $(CFLAGS) -ta=tesla -fast

clean:
	rm -f *.o *.vtk poissonACC_cpu poissonACC_CUDA
