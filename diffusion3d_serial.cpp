#include <iostream>
#include <algorithm>
#include <functional>
#include <iterator>
#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <cassert>
#include <vector>
#include <chrono>
#include <cmath>

typedef double value_type;
typedef std::size_t size_type;

class Diffusion3D
{

public:

  Diffusion3D(
          const value_type D, 
          const value_type rmax, 
          const value_type rmin,
          const size_type N
          )
    :
      D_(D)
    , rmax_(rmax)
    , rmin_(rmin)
    , N_(N)
  {
    N_tot = N_*N_*N_;

    //real space grid spacing
    dr_ = (rmax_ - rmin_) / (N_ - 1);

    //dt < dx*dx / (4*D) for stability
    dt_ = dr_ * dr_ / (6 * D_);

    //stencil factor
    fac_ = dt_ * D_ / (dr_ * dr_);

    rho_ = new value_type[N_tot];
    rho_tmp = new value_type[N_tot];

    std::fill(rho_, rho_+N_tot,0.0);
    std::fill(rho_tmp, rho_tmp+N_tot,0.0);

    InitializeSystem();
  }

  ~Diffusion3D()
  {
    delete[] rho_;
    delete[] rho_tmp;
  }

  void PropagateDensity();

  value_type GetSize() {
    value_type sum = 0;

    for(size_type i = 0; i < N_tot; ++i)
      sum += rho_[i];

    return dr_*dr_*sum;
  }

  value_type GetMoment() {
    value_type sum = 0;

    for(size_type i = 0; i < N_; ++i)
      for(size_type j = 0; j < N_; ++j) {
		for(size_type k = 0; k < N_; ++k) {
			value_type x = j*dr_ + rmin_;
			value_type y = i*dr_ + rmin_;
			value_type z = k*dr_ + rmin_;
			sum += rho_[i*N_ *(N_ - 1)+ j*(N_ - 1) + k] * (x*x + y*y + z*z);
			}
      }

    return dr_*dr_*sum;
  }
  void InitializeSystem();

  value_type GetTime() const {return time_;}

  void WriteDensity(const std::string file_name) const;

private:

  const value_type D_, rmax_, rmin_;
  const size_type N_;
  size_type N_tot;

  value_type dr_, dt_, fac_;

  value_type time_;

  value_type *rho_, *rho_tmp;
};

void Diffusion3D::PropagateDensity()
{  
  using std::swap;
  //Dirichlet boundaries; central differences in space, forward Euler
  //in time

  for(size_type i = 0; i < N_; ++i)
    for(size_type j = 0; j < N_; ++j)
		for(size_type k = 0; k < N_; ++k)
			rho_tmp[i*N_*(N_ -1) + j*(N_ - 1) + k] = 
			rho_[i*N_*(N_ -1) + j*(N_ - 1) + k]
			+
			fac_ 
			* 
			(
			 (j == N_-1 ? 0 : rho_[i*N_*(N_ -1) + (j+1)*(N_ -1) + k])
			 +
			 (j == 0 ? 0 : rho_[i*N_*(N_ -1) + (j-1)*(N_ -1) + k])
			 +
			 (i == N_-1 ? 0 : rho_[(i+1)*N_*(N_ -1) + j*(N_ -1) + k])
			 +
			 (i == 0 ? 0 : rho_[(i-1)*N_*(N_ -1) + j*(N_ -1) + k])
			 +
			 (k == N_-1 ? 0 : rho_[i*N_*(N_ -1) + j*(N_ -1) + (k+1)])
			 +
			 (k == 0 ? 0 : rho_[i*N_*(N_ -1) + j*(N_ -1) + (k-1)])
			 -
			 6*rho_[i*N_*(N_ -1) + j*(N_ - 1) + k]
			 );

  swap(rho_tmp,rho_);

  time_ += dt_;
}

void Diffusion3D::InitializeSystem()
{
  time_ = 0;

  //initialize rho(x,y,z,t=0)
  value_type bound = 1./2;

  for(size_type i = 0; i < N_; ++i)
    for(size_type j = 0; j < N_; ++j){
		for(size_type k = 0; k < N_; ++k){
		  if(std::fabs(i*dr_+rmin_) < bound && std::fabs(j*dr_+rmin_) < bound && std::fabs(k*dr_+rmin_) < bound){
			rho_[i*N_*(N_ -1) + j*(N_ -1) + k] = 1.0;
		  }
		  else{
			rho_[i*N_*(N_ -1) + j*(N_ -1) + k] = 0.0;
		  }
		}
    }
}

int main(int argc, char* argv[])
{
  assert(argc == 2);

  const value_type D = 1;
  const value_type tmax = 0.001;
  const value_type rmax = 1;
  const value_type rmin = -1;

  const size_type N_ = 1<<std::stoul(argv[1]);
  
  Diffusion3D System(D, rmax, rmin, N_);
  System.InitializeSystem();

  value_type time = 0;

  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  start = std::chrono::high_resolution_clock::now();
  
  const size_type max_steps = 20;


  for(size_type steps = 0; steps < max_steps; ++steps){
    System.PropagateDensity();
    time = System.GetTime();
    std::cout << time << '\t' << System.GetSize() << '\t' << System.GetMoment() << std::endl;
  }

  end = std::chrono::high_resolution_clock::now();

  double elapsed = std::chrono::duration<double>(end-start).count();

  std::cout << N_ << '\t' << elapsed << std::endl;

  return 0;
}
