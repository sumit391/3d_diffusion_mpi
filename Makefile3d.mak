## module load gcc open_mpi

CXX= mpicxx
CXXFLAGS?=-std=c++11 -O3 -DNDEBUG -fopenmp 

diffusion3d_mpi: diffusion3d_mpi_20140709.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
