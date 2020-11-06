#ifndef DETERMINANTAL_PP_HPP
#define DETERMINANTAL_PP_HPP

#include "base_pp.hpp"

class DeterminantalPP : public BasePP {
 protected:
  // truncation of the eigendecomposition
  int N;

  // parameters in the PES density
  double rho;
  double nu;
  double s; //in (0,1)

  // eigendecomposition params
  VectorXd phis;
  VectorXd phi_tildes;
  double Ds;
  MatrixXd Kappas;

  // Affine transformation to the unit square
  MatrixXd A;
  VectorXd b;

 public:
  DeterminantalPP() {std::cout << "initializing from scratch" << std::endl;}

  DeterminantalPP(int N, double rho, double nu, double s) : N(N), rho(rho), nu(nu), s(s) {}

  ~DeterminantalPP() {}

  void initialize() override;

  double dens(const MatrixXd &x, bool log = true) override;

  double papangelou(MatrixXd xi, const MatrixXd &x, bool log = true) override;

  double papangelou(const Point &xi, const std::list<Point> &x,
                    bool log = true) override;

  VectorXd phi_star_rng() override;

  double phi_star_dens(VectorXd xi, bool log = true) override;

  void update_hypers(const MatrixXd &active,
                     const MatrixXd &non_active);

  void get_state_as_proto(google::protobuf::Message *out) {}

  double rejection_sampling_M(int npoints) override;

  // NB assumes that X has been rescaled in -0.5, 0.5!
  double log_det_Ctilde(const MatrixXd& x);

  void eigen_decomposition();

  double estimate_mean_proposal_sigma() {return 2.0;}

  MatrixXd get_kappas() const {return Kappas;}
};

#endif
