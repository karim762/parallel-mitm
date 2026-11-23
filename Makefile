MPICC  ?= mpicc
CC     ?= gcc
CFLAGS ?= -O3 -march=native -Wall -Wextra -fopenmp

all: build/mitm build/gen_instance

build/mitm: src/mitm.c src/dict.h src/exchange.h src/speck.h
	@mkdir -p build
	$(MPICC) $(CFLAGS) src/mitm.c -o $@

build/gen_instance: tools/gen_instance.c src/speck.h
	@mkdir -p build
	$(CC) -O2 -Wall -Wextra tools/gen_instance.c -o $@

test: all
	./tests/run_tests.sh

clean:
	rm -rf build

.PHONY: all test clean
