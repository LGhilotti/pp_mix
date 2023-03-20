#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;

int main(){

  MatrixXd lamb (3,2);
  lamb << 3,0,
      1,0.1,
      4,-2;

  MatrixXd dat(2,3);
  dat << 10,2,-3,
        -4,-5,0;
  std::cout<<"dati: "<<dat<<std::endl;

  MatrixXd B ((dat*lamb).transpose());
  LLT<MatrixXd> A (lamb.transpose() * lamb);
  MatrixXd sol(A.solve(B)); // IT WORKS!
  std::cout<<"solutions: "<<sol<<std::endl;

  std::cout<<"sol_1dato: "<<A.solve(B.col(0))<<std::endl;
  std::cout<<"sol_2dato: "<<A.solve(B.col(1))<<std::endl;

  return 0;

}
