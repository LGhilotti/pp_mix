#include "../point_process/determinantalPP.hpp"
#include <vector>
using std::vector;

int main(){

  MatrixXd rect (2,2);
  rect << -4,-4,4,4;
  std::cout << "rect: \n" << rect<< std::endl;

  MatrixXd lamb (3,2);
  lamb << 10,0,
          10,0.1,
          10,0;

  LLT<MatrixXd> A (lamb.transpose() * lamb);
  double det = std::pow(A.matrixL().determinant(),2);
  std::cout<< "determinatnt: "<<det<<std::endl;


  DeterminantalPP dpp (rect, 1000, 6, 0.8, &lamb);
  Vector2d eta1(2,2);
  Vector2d eta2(3,2);
  Vector2d eta3(2,3);

  MatrixXd conf1(2,2);
  conf1 << eta1.transpose(), eta2.transpose();
  std::cout<<"conf1= "<<conf1<<std::endl;

  MatrixXd conf2(2,2);
  conf2 << eta1.transpose(), eta3.transpose();
  std::cout<<"conf2= "<<conf2<<std::endl;

  MatrixXd conf3(2,2);
  conf3 << eta2.transpose(), eta3.transpose();
  std::cout<<"conf3= "<<conf3<<std::endl;

  std::cout<<"density conf1= "<<dpp.dens(conf1,false)<<std::endl;

  std::cout<<"density conf2= "<<dpp.dens(conf2,false)<<std::endl;

  std::cout<<"density conf3= "<<dpp.dens(conf3,false)<<std::endl;


  //test sample_given_active
  MatrixXd act(4,2);
  act << 1,1,
        3,-2,
        1.5,0.8,
        2,-0.2;
  std::cout<<"defined act"<<std::endl;
  MatrixXd non_act(0,2);
  std::cout<<"defined non_act"<<std::endl;
  dpp.sample_given_active(&non_act,act,2.0);
  std::cout<<"non active: "<<non_act<<std::endl;

  return 0;

}
