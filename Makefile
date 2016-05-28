CC       = mpicc
CXX      = mpic++
ALLFLAGS = -Wall -g -DDEBUG=1
CFLAGS   = $(ALLFLAGS) -std=c99
CXXFLAGS = $(ALLFLAGS) -std=c++11
LDFLAGS  = $(ALLFLAGS) -std=c++11
MATMUL   = matrixmul
DEPS     = densematgen.o matrix_utils.o matrix.o utils.o
DEPFILE  = .dependencies

all:
	$(MAKE) $(MAKEOPTS) $(DEPFILE)
	$(MAKE) $(MAKEOPTS) $(MATMUL)

$(DEPFILE): *.cpp *.hpp *.c *.h
	$(CXX) -MM *.cpp *.c > $@

include $(DEPFILE)

$(MATMUL): $(MATMUL).o $(DEPS)
	$(CXX) $(LDFLAGS) $^ -o $@

%.o: %.c %.h
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp %.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o *core *~ *.out *.err $(MATMUL)
