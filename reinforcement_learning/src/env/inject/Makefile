CC     = gcc
CFLAGS = -shared -fPIC -O2 -Wall
LIBS   = -ldl -lrt $(shell sdl2-config --libs)
INC    = $(shell sdl2-config --cflags)

all: tw_input_hook.so

tw_input_hook.so: tw_input_hook.c tw_shm.h
	$(CC) $(CFLAGS) $(INC) -o $@ tw_input_hook.c $(LIBS)

clean:
	rm -f tw_input_hook.so