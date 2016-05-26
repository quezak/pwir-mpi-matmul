CC       = mpicc
CXX      = mpic++
CFLAGS   = -Wall -std=c99 -O3
CXXFLAGS = -Wall -std=c++11 -O3
LDFLAGS  = -Wall -O3 -lstdc++
MATMUL   = matrixmul test
DEPS     = densematgen.o matrix_utils.o

all: $(MATMUL)

$(TEST): $(TEST).o $(DEPS)

$(MATMUL): $(MATMUL).o $(DEPS)
	$(CXX) $(LDFLAGS) $^ -o $@

clean:
	rm -f *.o *core *~ *.out *.err $(ALL)
