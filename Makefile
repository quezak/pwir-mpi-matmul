CC       = mpicc
CXX      = mpic++
CFLAGS   = -Wall -std=c99 -O3
CXXFLAGS = -Wall -std=c++11 -O3
LDFLAGS  = -Wall -O3 -lstdc++
MATMUL   = matrixmul
DEPS     = densematgen.o

all: $(MATMUL)

$(MATMUL): $(MATMUL).o $(DEPS)
	$(CXX) $(LDFLAGS) $^ -o $@

clean:
	rm -f *.o *core *~ *.out *.err $(ALL)
