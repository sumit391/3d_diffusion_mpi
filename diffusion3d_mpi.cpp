//For the solutionm of 3Dimensional Diffusion Equation
// This would take one variable input which is the grid size in a Dimension (N_ Value)
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
#include <cmath>
#include <mpi.h>
#include <omp.h>

typedef double value_type;
typedef std::size_t size_type;

#define TAG 0
  
struct world_info
{
  int size;
  int dims_x;
  int dims_y;
  int dims_z;

  int left_proc;
  int right_proc;
  int top_proc;
  int bottom_proc;
  int front_proc;
  int back_proc;

  int rank;
  int cart_rank;
  int coord_x;
  int coord_y;
  int coord_z;

} world;

struct position
{
  value_type x,y,z;
};

class Diffusion3D
{
  struct position_ext
  {
    value_type x,y,z,rho;
  };

public:

  Diffusion3D(
	      const value_type d, 
	      const value_type rmax, 
	      const value_type rmin,
	      const size_type N
	      )
    :
    d_(d)
    , rmax_(rmax)
    , rmin_(rmin)
  {
    //global grid
    Nx_glo = N;
    Ny_glo = N;
    Nz_glo = N;
    NN_glo = Nx_glo * Ny_glo * Nz_glo;

    //local grid
    Nx_loc = Nx_glo / world.dims_x;
    Ny_loc = Ny_glo / world.dims_y;
    Nz_loc = Nz_glo / world.dims_z;
    NN_loc = Nx_loc * Ny_loc * Nz_loc;

    //build periodic process geometry with Cartesian Communicator
    int periods[3] = {false, false, false};
    int dims[3] = {world.dims_x, world.dims_y, world.dims_z};

    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, true, &cart_comm);

    MPI_Comm_rank(cart_comm,&world.cart_rank);

    MPI_Cart_shift(cart_comm, 0, 1, &world.left_proc, &world.right_proc);
    MPI_Cart_shift(cart_comm, 1, 1, &world.bottom_proc, &world.top_proc);
    MPI_Cart_shift(cart_comm, 2, 1, &world.back_proc, &world.front_proc);
  
    int coords[3];
    MPI_Cart_coords(cart_comm, world.cart_rank, 3, coords);

    world.coord_x = coords[0];
    world.coord_y = coords[1];
    world.coord_z = coords[3];

    //build contiguous (rows) and strided vectors (columns) for boundaries. each process has a square tile in
    //the cartesian grid


    MPI_Type_vector(Nz_loc, Nx_loc,(Nx_loc+2)* (Ny_loc +1 ), MPI_DOUBLE, &bottom_boundary);
    MPI_Type_commit(&bottom_boundary);

    MPI_Type_vector(Nz_loc, Nx_loc,(Nx_loc+2)*( Ny_loc +1), MPI_DOUBLE, &top_boundary);
    MPI_Type_commit(&top_boundary);
	
    MPI_Type_vector(Ny_loc*Nz_loc, 1, Nx_loc + 2, MPI_DOUBLE, &left_boundary);
    MPI_Type_commit(&left_boundary);

    MPI_Type_vector(Ny_loc*Nz_loc, 1, Nx_loc + 2, MPI_DOUBLE, &right_boundary);
    MPI_Type_commit(&right_boundary);
	
    MPI_Type_vector(Ny_loc, Nx_loc,Nx_loc + 2, MPI_DOUBLE, &back_boundary);
    MPI_Type_commit(&back_boundary);

    MPI_Type_vector(Ny_loc, Nx_loc,Nx_loc + 2, MPI_DOUBLE, &front_boundary);
    MPI_Type_commit(&front_boundary);

    //real space grid spacing
    dr_ = (rmax_ - rmin_) / (N - 1);

    //sub-domain boundaries
    xmin_loc = rmin + world.coord_x * Nx_loc * dr_;
    xmax_loc = xmin_loc + (Nx_loc - 1) * dr_;
    ymin_loc = rmin + world.coord_y * Ny_loc * dr_;
    ymax_loc = ymin_loc + (Ny_loc - 1) * dr_;
    zmin_loc = rmin + world.coord_z * Nz_loc * dr_;
    zmax_loc = zmin_loc + (Nz_loc - 1) * dr_;

    //dt < dx*dx / (4*d) for stability
    dt_ = dr_ * dr_ / (6 * d_);

    //stencil factor
    fac_ = dt_ * d_ / (dr_ * dr_);

    //allocate and fill density
    rho_ = new value_type[(Nx_loc + 2) * (Ny_loc + 2) * (Nz_loc + 2)];
    rho_tmp = new value_type[(Nx_loc + 2) * (Ny_loc + 2) * (Nz_loc + 2)];

    std::fill(rho_, rho_ + ((Nx_loc + 2) * (Ny_loc + 2) * (Nz_loc + 2)), 0.0);
    std::fill(rho_tmp, rho_tmp + ((Nx_loc + 2) * (Ny_loc + 2) * (Nz_loc + 2)), 0.0);
  }
  
  //destructor
  ~Diffusion3D()
  {
    delete[] rho_;
    delete[] rho_tmp;

    MPI_Type_free(&left_boundary);
    MPI_Type_free(&right_boundary);
    MPI_Type_free(&bottom_boundary);
    MPI_Type_free(&top_boundary);
    MPI_Type_free(&back_boundary);
    MPI_Type_free(&front_boundary);

    MPI_Comm_free(&cart_comm);
  }
  
  void initialize();
  void propagate_density();
  
  inline position get_position(size_type i, size_type j, size_type k) const;

  value_type get_size();
  value_type get_moment();
  value_type get_time() const {return time_;}

  void write_density(const std::string file_name) const;

private:

  inline void stencil(const size_type i, const size_type j, const size_type k);

  const value_type d_, rmax_, rmin_;

  value_type xmin_loc, ymin_loc, zmin_loc;
  value_type xmax_loc, ymax_loc, zmax_loc;

  int NN_loc, Nx_loc, Ny_loc, Nz_loc;
  int NN_glo, Nx_glo, Ny_glo, Nz_glo;

  value_type dr_, dt_, fac_;

  value_type time_;

  value_type *rho_, *rho_tmp;

  MPI_Datatype left_boundary, right_boundary, bottom_boundary, top_boundary, back_boundary, front_boundary;;

  MPI_Comm cart_comm;
};

inline position Diffusion3D::get_position(size_type i, size_type j, size_type k) const
{
  position p;
  p.x = xmin_loc + j*dr_;
  p.y = ymin_loc + i*dr_;
  p.z = zmin_loc + i*dr_;
  return p;
}

value_type Diffusion3D::get_size() 
{
  value_type sum_local = 0.0;
  #pragma omp parallel for reduction(+:sum_local)
  for(int i = 0; i < Nz_loc; ++i){
	for(int j = 0; j < Ny_loc; ++j){
		for(int k = 0; k < Nx_loc; ++k){
		const size_type ind_rho = (i+1)*(Nx_loc+2)*(Ny_loc+2) + (j+1)*(Nx_loc+2) + (k+1);
		sum_local += rho_[ind_rho];
		}
	}
  }
  //fetch partial sum from all processes
  value_type sum = 0.0;
  MPI_Reduce(&sum_local, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);

  return dr_*dr_*sum;
}

value_type Diffusion3D::get_moment() 
{
  value_type sum_local = 0.0;
  #pragma omp parallel for reduction(+:sum_local)
  for(int i = 0; i < Nz_loc; ++i)
    for(int j = 0; j < Ny_loc; ++j){
		for(int k = 0; k < Nx_loc; ++k){
			const size_type ind_rho = (i+1)*(Nx_loc+2)*(Ny_loc+2) + (j+1)*(Nx_loc+2) + (k+1);
			const position p = get_position(i,j,k);
			sum_local += rho_[ind_rho] * (p.x * p.x + p.y * p.y + p.z * p.z);
		}
	}

  //fetch partial sum from all processes
  value_type sum = 0.0;
  MPI_Reduce(&sum_local, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);

  return dr_*dr_*sum;
}

inline void Diffusion3D::stencil(const size_type i, const size_type j, const size_type k) 
{
  const size_type ind = i*(Nx_loc+2)*(Ny_loc+2) + j*(Nx_loc+2) + k;

  //FTCS stencil
  rho_tmp[ind] = 
    rho_[ind]
    +
    fac_ 
    * 
    (
     rho_[i*(Nx_loc+2)*(Ny_loc+2) + (j+1)*(Nx_loc+2) + k]
     +
     rho_[i*(Nx_loc+2)*(Ny_loc+2) + (j-1)*(Nx_loc+2) + k]
     +
     rho_[(i+1)*(Nx_loc+2)*(Ny_loc+2) + j*(Nx_loc+2) + k]
     +
     rho_[(i-1)*(Nx_loc+2)*(Ny_loc+2) + j*(Nx_loc+2) + k]
     +
     rho_[(i)*(Nx_loc+2)*(Ny_loc+2) + j*(Nx_loc+2) + (k+1)]
     +
     rho_[(i)*(Nx_loc+2)*(Ny_loc+2) + j*(Nx_loc+2) + (k-1)]
     -
     6*rho_[ind]
     );
}

//Propagation of Density
void Diffusion3D::propagate_density()
{  
  MPI_Request reqs[12];
  MPI_Status status[12];

  //exchange boundaries along x-direction
  if(world.coord_x % 2 == 0){

    MPI_Isend(&rho_[((Nx_loc+2)*(Ny_loc+2))+(Nx_loc+2)+1],1,left_boundary,world.left_proc,TAG,cart_comm,&reqs[0]);
    MPI_Irecv(&rho_[((Nx_loc+2)*(Ny_loc+2))+Nx_loc+2],1,left_boundary,world.left_proc,TAG,cart_comm,&reqs[1]);
    MPI_Isend(&rho_[((Nx_loc+2)*(Ny_loc+2))+(Nx_loc+2)+Nx_loc],1,right_boundary,world.right_proc,TAG,cart_comm,&reqs[2]);
    MPI_Irecv(&rho_[((Nx_loc+2)*(Ny_loc+2))+(Nx_loc+2)+Nx_loc+1],1,right_boundary,world.right_proc,TAG,cart_comm,&reqs[3]);

  }
  else{

    MPI_Irecv(&rho_[((Nx_loc+2)*(Ny_loc+2))+(Nx_loc+2)+Nx_loc+1],1,right_boundary,world.right_proc,TAG,cart_comm,&reqs[0]);
    MPI_Isend(&rho_[((Nx_loc+2)*(Ny_loc+2))+(Nx_loc+2)+Nx_loc],1,right_boundary,world.right_proc,TAG,cart_comm,&reqs[1]);
    MPI_Irecv(&rho_[((Nx_loc+2)*(Ny_loc+2))+Nx_loc+2],1,left_boundary,world.left_proc,TAG,cart_comm,&reqs[2]);
    MPI_Isend(&rho_[((Nx_loc+2)*(Ny_loc+2))+(Nx_loc+2)+1],1,left_boundary,world.left_proc,TAG,cart_comm,&reqs[3]);

  }

  //exchange boundaries along y-direction
  if(world.coord_y % 2 == 0){

    MPI_Isend(&rho_[(Nx_loc+2)*(Ny_loc+2)+(Nx_loc+2)+1],1,bottom_boundary,world.bottom_proc,TAG,cart_comm,&reqs[4]);
    MPI_Irecv(&rho_[(Nx_loc+2)*(Ny_loc+2)+1],1,bottom_boundary,world.bottom_proc,TAG,cart_comm,&reqs[5]);
    MPI_Isend(&rho_[(Nx_loc+2)*(Ny_loc+(Ny_loc+2))+1],1,top_boundary,world.top_proc,TAG,cart_comm,&reqs[6]);
    MPI_Irecv(&rho_[(Nx_loc+2)*((Ny_loc+1)+(Ny_loc+2))+1],1,top_boundary,world.top_proc,TAG,cart_comm,&reqs[7]);

  }
  else{

    MPI_Irecv(&rho_[(Nx_loc+2)*((Ny_loc+1)+(Ny_loc+2))+1],1,top_boundary,world.top_proc,TAG,cart_comm,&reqs[4]);
    MPI_Isend(&rho_[(Nx_loc+2)*(Ny_loc+(Ny_loc+2))+1],1,top_boundary,world.top_proc,TAG,cart_comm,&reqs[5]);
    MPI_Irecv(&rho_[(Nx_loc+2)*(Ny_loc+2)+1],1,bottom_boundary,world.bottom_proc,TAG,cart_comm,&reqs[6]);
    MPI_Isend(&rho_[(Nx_loc+2)*(Ny_loc+2)+(Nx_loc+2)+1],1,bottom_boundary,world.bottom_proc,TAG,cart_comm,&reqs[7]);

  }

   //exchange boundaries along z-direction 
  if(world.coord_z % 2 == 0){

    MPI_Isend(&rho_[(Nx_loc+2)*(Ny_loc+2)+(Nx_loc+2)+1],1,back_boundary,world.back_proc,TAG,cart_comm,&reqs[8]);
    MPI_Irecv(&rho_[(Nx_loc+2)+1],1,back_boundary,world.back_proc,TAG,cart_comm,&reqs[9]);
    MPI_Isend(&rho_[(Nx_loc+2)*(Ny_loc+2)*(Nz_loc)+1],1,front_boundary,world.front_proc,TAG,cart_comm,&reqs[10]);
    MPI_Irecv(&rho_[(Nx_loc+2)*(Ny_loc+2)*(Nz_loc+1)+1],1,front_boundary,world.front_proc,TAG,cart_comm,&reqs[11]);

  }
  else{

    MPI_Irecv(&rho_[(Nx_loc+2)*(Ny_loc+2)*(Nz_loc+1)+1],1,front_boundary,world.front_proc,TAG,cart_comm,&reqs[8]);
    MPI_Isend(&rho_[(Nx_loc+2)*(Ny_loc+2)*(Nz_loc)+1],1,front_boundary,world.front_proc,TAG,cart_comm,&reqs[9]);
    MPI_Irecv(&rho_[(Nx_loc+2)+1],1,back_boundary,world.back_proc,TAG,cart_comm,&reqs[10]);
    MPI_Isend(&rho_[(Nx_loc+2)*(Ny_loc+2)+(Nx_loc+2)+1],1,back_boundary,world.back_proc,TAG,cart_comm,&reqs[11]);
	
  }
  
  //update interior of sub-domain
  #pragma omp parallel for
  for(int i = 2; i < Nz_loc; ++i)
    for(int j = 2; j < Ny_loc; ++j)
		for(int k = 2; k < Nx_loc; ++k)
			stencil(k,j,i);

  //ensure boundaries have arrived
  MPI_Waitall(12,reqs,status);

  //update the rest
  #pragma omp parallel for
  for(int j = 1; j < Ny_loc; ++j){
		for(int k = 1; k < Nz_loc; ++k){
			stencil(1,j,k); //left column
			stencil(Nx_loc,j,k); //right column
		}
	}
  #pragma omp parallel for
  for(int i = 1; i < Nx_loc; ++i){
		for(int k = 1; k < Nz_loc; ++k){
			stencil(i,1,k); //bottom row
			stencil(i,Ny_loc,k); //top row
		}
	}
  #pragma omp parallel for
  for(int i = 1; i < Nx_loc; ++i){
		for(int j = 1; j < Ny_loc; ++j){
			stencil(i,j,1); //back row
			stencil(i,j,Nz_loc); //front row
		}
	}

  //assign new density
  using std::swap;
  swap(rho_tmp,rho_);

  time_ += dt_;
}

void Diffusion3D::initialize()
{
  //initialize grid in non-overlapping sub-domains
  
  time_ = 0;

  const value_type bound = 1./2;

  //initialize rho(x,y,z,t=0)

  for(int i = 0; i < Nz_loc; ++i)
    for(int j = 0; j < Ny_loc; ++j){
		for(int k = 0; k < Nx_loc; ++k){
			const size_type ind_rho = (i+1)*(Nx_loc+2)*(Ny_loc+2) + (j+1)*(Nx_loc+2) + (k+1);
			const position p = get_position(i,j,k);

			if(std::fabs(p.x) < bound && std::fabs(p.y) < bound && std::fabs(p.z) < bound){
				rho_[ind_rho] = 1;
			}
			else{
				rho_[ind_rho] = 0;
			}
		}
    }
}

int main(int argc, char* argv[])
{
  //initialize MPI domain
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided ); //MPI with OpemMP threads 
  assert(provided >= MPI_THREAD_MULTIPLE);

  
  MPI_Comm_size(MPI_COMM_WORLD, &world.size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world.rank);

  int dims[3] = {0,0,0};
  MPI_Dims_create(world.size, 3, dims);
  world.dims_x = dims[0];
  world.dims_y = dims[1];
  world.dims_z = dims[2];
  
  //Calculate the number of threads
  int nrthr = 0;  
  #pragma omp parallel
  {
  #pragma omp single
    {
     nrthr = omp_get_num_threads( );
    }
  }

  if(world.rank == 0)
    std::cout 
      << "processes: " << world.size << "\n" 
      << "dims_x: " << world.dims_x << "\n"
      << "dims_y: " << world.dims_y << "\n"
      << "dims_z: " << world.dims_z << "\n"
      << "thread: " << nrthr << "\n"
      << std::endl;

  //params

  assert(argc == 2);

  //get the desired N value
  const size_type tmp = atoi(argv[1]);

  //make gridpoints multiple of procs in all directions
  const size_type N_ = tmp % world.size == 0 ? tmp : tmp + (world.size - tmp % world.size);
  assert(N_ % world.size == 0);
  //std::cout << "N_ val after checking constraint" << N_ << std::endl;
  const value_type d_ = 1;
  const value_type tmax = 0.0002;
  const value_type rmax = 1;
  const value_type rmin = -1;

  if(world.rank == 0)
    std::cout 
      << "domain: " << N_ << " x " << N_ << " x " << N_ << "\n"
      << "diffusion coefficient: " << d_ << "\n"
      << "tmax: " << tmax << "\n"
      << "rmax: " << rmax << "\n"
      << "rmin: " << rmin << "\n"
      << "Input Arg (tmp): " << tmp << "\n"
      << "Input actual Arg: " << argv[1] << "\n"
      << std::endl;
  
  {
    Diffusion3D system(d_, rmax, rmin, N_);
    system.initialize();

    MPI_Barrier(MPI_COMM_WORLD);

    value_type time = 0;
    const size_type max_steps = 20;

    double start = MPI_Wtime();

    
	//#pragma omp parallel for
	{
		for(size_type steps = 0; steps < max_steps; ++steps){
			system.propagate_density();
			time = system.get_time();
			value_type sys_size = system.get_size();
			value_type sys_moment = system.get_moment();
			//#pragma omp atomic
                         {
			   if(world.rank == 0) std::cout << time << '\t' << sys_size << '\t' << sys_moment << std::endl;
                         }
		}
	}

    double end = MPI_Wtime();

    double elapsed = end-start;


    if(world.rank == 0){

      std::string sep = "_";

      std::cout
	<< "performance: " << '\t'
	<< world.size << '\t'
	<< elapsed << '\t'
	<< N_ << '\t'
	<< "\n*********\n"
	<< std::endl;

    }

  }//destroy system before finalizing  

  MPI_Finalize();

  return 0;
}
