#ifndef UNI_FACTOR_DPP_HPP
#define UNI_FACTOR_DPP_HPP

#include "base_determinantalPP.hpp"

class UniDpp: public BaseDeterminantalPP {

public:

  UniDpp(const MatrixXd &ranges, int N, double c, double s): BaseDeterminantalPP(ranges,N,c,s) {};

  // set the pointer to Lambda and performs the initial decomposition
  void set_decomposition(const MatrixXd * lambda) override;

  // computes the decomposition (Ds, phis, phi_tildes, c_star) of the dpp
  void compute_eigen_and_cstar();

  void compute_Kappas() override; // compute just once the grid for summation over Z

};

#endif
