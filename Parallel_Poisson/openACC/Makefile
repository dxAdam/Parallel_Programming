CC=pgcc
CFLAGS=-acc -Minfo=acc

all: poissonACC_cpu poissonACC_gpu

poissonACC_cpu: poissonACC.c vtk.c
	$(CC) -o poissonACC_cpu poissonACC.c vtk.c $(CFLAGS) -ta=multicore -lm

poissonACC_gpu: poissonACC.c vtk.c
	$(CC) -o poissonACC_gpu poissonACC.c vtk.c $(CFLAGS) -ta=tesla

clean:
	rm -f *.o *.vtk poissonACC_cpu poissonACC_gpu
