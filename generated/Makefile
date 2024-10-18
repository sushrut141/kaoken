CC = clang
CFLAGS = -arch arm64 -O3

all: main

main: $(wildcard *.o)
	$(CC) $(CFLAGS) -o main $(wildcard *.c)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean :
	-rm main         