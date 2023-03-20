#include "../rng.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <vector>
using namespace Eigen;
using namespace stan::math;

int main(){

  VectorXd m1(2);
  m1<<1,2;
  std::cout<<"Vector mean m1: "<<m1<<std::endl;
  MatrixXd prec(2,2);
  prec<<1,0,
        0,2;

  std::cout<<"Matrix prec: "<<prec<<std::endl;

  VectorXd sample = multi_normal_prec_rng(m1,prec,Rng::Instance().get());
  std::cout<<"sampled value: "<<sample<<std::endl;
  //////////////
  VectorXd m2(2);
  m2<<8,2;
  std::cout<<"Vector mean m2: "<<m2<<std::endl;
  MatrixXd v(2,2);
  v<<m1,m2;
  std::cout<<"Matrix mean v: "<<v<<std::endl;

  MatrixXd sol=multi_normal_prec_rng(v,prec,Rng::Instance().get());
  std::cout<<"sampled values: "<<sol<<std::endl;

  /// CANNOT SAMPLE SIMULTANEOUSLY WITH multi_normal_prec_rng
  return 0;

}
