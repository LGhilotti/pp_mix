#include "covmat.hpp"

CovMat::CovMat(MatrixXd cov): cov(cov) {
    cho_factor = LLT<MatrixXd>(cov);
    cho_factor_eval = cho_factor.matrixL();
    VectorXd diag = cho_factor_eval.diagonal();
    log_det = 2 * log(diag.array()).sum();
}

CovMat::CovMat(const CovMat &other)
{   
    this->cov = other.cov;
    this->cho_factor = other.cho_factor;
    this->cho_factor_eval = other.cho_factor_eval;
    this->log_det = other.log_det;
}

MatrixXd CovMat::get_cov() const {
    return cov;
}

LLT<MatrixXd> CovMat::get_cho_factor() const
{
    return cho_factor;
}

MatrixXd CovMat::get_cho_factor_eval() const
{
    return cho_factor_eval;
}

double CovMat::get_log_det() const {
    return log_det;
}

PrecMat::PrecMat(const MatrixXd &prec): prec(prec) {
    cho_factor = LLT<MatrixXd>(prec);
    cho_factor_eval = cho_factor.matrixL();
    const VectorXd& diag = cho_factor_eval.diagonal();
    log_det = 2 * log(diag.array()).sum();
}

MatrixXd PrecMat::get_prec() const
{
    return prec;
}

LLT<MatrixXd> PrecMat::get_cho_factor() const
{
    return cho_factor;
}

const MatrixXd& PrecMat::get_cho_factor_eval() const
{
    return cho_factor_eval;
}

double PrecMat::get_log_det() const
{
    return log_det;
}