#include <iostream>
#include <stan/math/prim.hpp>
#include <Eigen/Dense>
#include <vector>
#include "../rng.hpp"
using namespace Eigen;

int main(){

  VectorXd m(2);
  m << 1,2;
  std::cout<<"m: "<<m<<std::endl;

  VectorXd p(2);
  p << 0.1,0.2;
  std::cout<<"p: "<<p<<std::endl;

  std::vector<double> samples = stan::math::normal_rng(std::vector<double>(m.data(),m.data()+m.size()),
        std::vector<double>(p.data(),p.data()+p.size()),Rng::Instance().get());

  std::vector<double> samples2 = stan::math::gamma_rng(std::vector<double>(5,2.0),
          std::vector<double>(5,4.0), Rng::Instance().get());

  for (int i=0; i<samples2.size();i++)
  std::cout<<samples2[i]<<std::endl;

  return 0;

}
