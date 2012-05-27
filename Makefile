CFLAGS		+= -std=c99 -O6 -I/opt/local/cuda/include -W -Wall -fopenmp
LDFLAGS		:= -L/addons/cuda/NVIDIA-Linux-x86_64-270.41.19/
LDLIBS		:= -lOpenCL -lpthread -fopenmp

SOURCES		:= $(wildcard *.cl)
EXEC		:= $(SOURCES:.cl=)

target: $(EXEC)

stencil.o: constantes.h

clean:
	rm -rf $(EXEC) *.o
