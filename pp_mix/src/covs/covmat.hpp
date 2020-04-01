#ifndef COVMAT_HPP
#define COVMAT_HPP

#include <Eigen/Dense>
using namespace Eigen;

class CovMat {
 protected:
    MatrixXd cov;
    LLT<MatrixXd> cho_factor;
    MatrixXd cho_factor_eval;
    double log_det;

 public:
    CovMat() {}
    ~CovMat() {}

    CovMat(MatrixXd cov);

    CovMat(const CovMat &other);

    MatrixXd get_cov() const;

    LLT<MatrixXd> get_cho_factor() const;

    MatrixXd get_cho_factor_eval() const;

    double get_log_det() const;
};


class PrecMat {
 protected:
   MatrixXd prec;
   LLT<MatrixXd> cho_factor;
   MatrixXd cho_factor_eval;
   double log_det;

 public:
   PrecMat() {}
   ~PrecMat() {}

   PrecMat(const MatrixXd &prec);

   MatrixXd get_prec() const;

   LLT<MatrixXd> get_cho_factor() const;

   const MatrixXd& get_cho_factor_eval() const;

   double get_log_det() const;
};

#endif 