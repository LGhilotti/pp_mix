#include "../precs/delta_wishart.hpp"

int main(){

  Delta_Wishart dw(3,2,1);
  std::cout<<"dof= "<<dw.get_df()<<std::endl;
  std::cout<<"psi matrix= "<<dw.get_psi()<<std::endl;
  std::cout<<dw.mean();
  //PrecMat mat=dw.sample_prior();
  std::cout<<"Sample from prior= "<<std::endl;
  std::cout<<dw.sample_prior()<<std::endl;
  Vector2d v1(1,3);
  Vector2d v2(-2,2);
  std::vector<VectorXd> vec{v1,v2};
  Vector2d mean(1,1);
  PrecMat second(dw.sample_alloc(vec,dw.sample_prior(),mean));
  std::cout<<"Sample alloc= "<<second<<std::endl;
  std::cout<<"log_density= "<<dw.lpdf(second)<<std::endl;
  return 0;
}
