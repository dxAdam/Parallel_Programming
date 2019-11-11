CC=gcc
CFLAGS=-std=c99 -lm
poissonSerial: poissonSerial.c vtk.c
	$(CC) -o poissonSerial poissonSerial.c vtk.c $(CFLAGS)

clean:
	rm -f *.o *.vtk poissonSerial
