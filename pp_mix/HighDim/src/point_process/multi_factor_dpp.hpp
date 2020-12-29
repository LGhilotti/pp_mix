#ifndef MULTI_FACTOR_DPP_HPP
#define MULTI_FACTOR_DPP_HPP

#include "base_determinantalPP.hpp"

class MultiDpp: public BaseDeterminantalPP {

private:

  const MatrixXd * Lambda; // const pointer to the Lambda matrix

  double c_star_tmp; // c* of X (DPP) wrt the last passed Lambda

  // eigendecomposition params for last passed Lambda
  VectorXd phis_tmp;
  VectorXd phi_tildes_tmp;
  double Ds_tmp;

public:

  MultiDpp(const MatrixXd &ranges, int N, double c, double s): BaseDeterminantalPP(ranges,N,c,s) {};

  // Sets the pointer to Lambda and sets phis, phi_tildes, Ds and c_star
  void compute_eigen_and_cstar(const MatrixXd * Lamb) override; // Once set these parameters, no more difference for multi/uni factor

  void compute_Kappas() override; // compute just once the grid for summation over Z^dim

};

#endif
