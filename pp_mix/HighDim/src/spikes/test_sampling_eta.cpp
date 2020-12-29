#include "../rng.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>
using namespace Eigen;
using namespace stan::math;

int main(){

  MatrixXd Lambda (3,2);
  Lambda << 3,0,
      1,0.1,
      4,-2;

  VectorXd sigma_bar(3);
  sigma_bar<<1,4,2;

  std::cout<<"Lambda: "<<Lambda<<std::endl;
  std::cout<<"sigma_bar: "<<sigma_bar<<std::endl;

  MatrixXd a_means(2,2);
  a_means<<3,4,
          -1,-1;

  MatrixXd p1(2,2);
  p1<<1,0,
      0,2;
  std::vector<MatrixXd> a_deltas{p1,p1};

  std::cout<<"precisions: "<<a_deltas[0]<<"\n"<<a_deltas[1]<<std::endl;

  MatrixXd data(2,3);
  data<<6,7,3,
        0,-2,-3;

  VectorXi v(2);
  v<<0,1;
  std::cout<<"vector v: "<<v<<std::endl;
  std::cout<<"first 2 columns: "<<data(Eigen::all,v)<<std::endl;

  int ndata=2;
  int dim_fact=2;
  std::vector<std::vector<int>> obs_by_clus(2);
  obs_by_clus[0].push_back(0);
  obs_by_clus[1].push_back(1);

  MatrixXd M0(Lambda.transpose() * sigma_bar.asDiagonal());
  MatrixXd M1( M0 * Lambda);
  std::vector<MatrixXd> Sn_bar(a_means.rows());
  // type LLT for solving systems of equations
  std::vector<LLT<MatrixXd>> Sn_bar_cho(a_means.rows());

  for (int i=0; i < a_means.rows(); i++){
    Sn_bar[i]=M1+a_deltas[i];
    Sn_bar_cho[i]= LLT<MatrixXd>(Sn_bar[i]);
  }

  MatrixXd M2(M0*data.transpose());
  MatrixXd G(ndata,dim_fact);
  // known terms of systems is depending on the single data
  // For cluster 1: I build the matrix of known terms
  for (int i=0; i < a_means.rows(); i++){
    MatrixXd B1(dim_fact,obs_by_clus[i].size());
    B1 = (a_deltas[i] * a_means.row(i).transpose()).replicate(1,B1.cols());
    B1 +=M2(all,obs_by_clus[i]);
    MatrixXd sol(Sn_bar_cho[i].solve(B1)); // each column of sol has solution for points in the cluster.
    std::cout<<"sol: "<<sol<<std::endl;
    G(obs_by_clus[i],all)=sol.transpose();
  }

  std::cout<<"gamma_n: "<<G<<std::endl;
  for (int i=0;i<a_means.rows();i++){
    std::cout<<"SnBar: "<<Sn_bar[i]<<std::endl;
  }
  return 0;
}
