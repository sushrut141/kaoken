CC = clang
CFLAGS = -arch arm64 -O3
CLANG_FORMAT = clang-format

all: main

main: $(wildcard *.o)
	$(CC) $(CFLAGS) -o main $(wildcard *.c)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

format: $(wildcard *.c)
	$(CLANG_FORMAT) -i *.c

clean :
	-rm main         