CC=pgcc
CFLAGS=-acc -fast -Minfo=all -Mprof=ccff

all: poissonACC_cpu poissonACC_gpu

poissonACC_cpu: poissonACC.c vtk.c
	$(CC) -o poissonACC_cpu poissonACC.c vtk.c $(CFLAGS) -ta=multicore -lm

poissonACC_gpu: poissonACC.c vtk.c
	$(CC) -o poissonACC_gpu poissonACC.c vtk.c $(CFLAGS) -ta=tesla:cc60

clean:
	rm -f *.o *.vtk poissonACC_cpu poissonACC_gpu
