#include "../jumps/jump_gamma.hpp"

int main(){

  double a=3,b=2;
  GammaJump gj (a,b);
  double lap = gj.laplace(2);
  std::cout<<"laplace= "<<lap<<std::endl;
  std::cout<<"sample_alloc= "<<gj.sample_alloc(10,2)<<std::endl;
  std::cout<<"sample_nonalloc= "<<gj.sample_nonalloc(2)<<std::endl;
  return 0;
}
