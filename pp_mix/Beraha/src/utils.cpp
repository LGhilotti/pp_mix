#include "utils.hpp"

void delete_row(MatrixXd *x, int ind) {
  int nrow = x->rows() - 1;
  int dim = x->cols();
  if (ind < nrow) {
    x->block(ind, 0, nrow - ind, dim) = x->block(ind + 1, 0, nrow - ind, dim);
  }

  x->conservativeResize(nrow, dim);
}

void delete_elem(VectorXd *x, int ind) {
  int size = x->size() - 1;
  if (ind < size) x->segment(ind, size - ind) = x->segment(ind + 1, size - ind);

  x->conservativeResize(size);
}

MatrixXd delete_row(const MatrixXd &x, int ind) {
  MatrixXd out = x;
  int nrow = x.rows() - 1;
  int dim = x.cols();
  if (ind < nrow) {
    out.block(ind, 0, nrow - ind, dim) = out.block(ind + 1, 0, nrow - ind, dim);
  }

  out.conservativeResize(nrow, dim);
  return out;
}

VectorXd delete_elem(const VectorXd &x, int ind) {
  VectorXd out = x;
  int size = x.size() - 1;
  if (ind < size)
    out.segment(ind, size - ind) = out.segment(ind + 1, size - ind);

  out.conservativeResize(size);
  return out;
}

MatrixXd vstack(const std::vector<VectorXd> &rows) {
  int nrows = rows.size();
  int ncols = rows[0].size();

  MatrixXd out(nrows, ncols);
  for (int i = 0; i < nrows; i++) out.row(i) = rows[i].transpose();

  return out;
}

double o_multi_normal_prec_lpdf(const VectorXd &x, const VectorXd &mu,
                                const PrecMat &sigma) {
  using stan::math::NEG_LOG_SQRT_TWO_PI;

  double out = 0.5 * sigma.get_log_det() + NEG_LOG_SQRT_TWO_PI * x.size();
  out -= 0.5 * (sigma.get_cho_factor_eval() * (x - mu)).squaredNorm();
  return out;
}

double o_multi_normal_prec_lpdf(const std::vector<VectorXd> &x,
                                const VectorXd &mu, const PrecMat &sigma) {
  using stan::math::NEG_LOG_SQRT_TWO_PI;

  int n = x.size();
  double out = sigma.get_log_det() * n;

  const MatrixXd &cho_sigma = sigma.get_cho_factor_eval();

  std::vector<double> loglikes(n);
  for (int i = 0; i < n; i++) {
    loglikes[i] = (cho_sigma * (x[i] - mu)).squaredNorm();
  }

  out -= std::accumulate(loglikes.begin(), loglikes.end(), 0.0);

  return 0.5 * out;
}

// generate from truncated normal by rejection sampling
// !! might not be the best idea
double trunc_normal_rng(double mu, double sigma, double lower, double upper,
                        std::mt19937_64 &rng) {
  while (true) {
    double val = stan::math::normal_rng(mu, sigma, rng);
    if (val <= upper && val >= lower) return val;
  }
}

double trunc_normal_rng_inversion(double mu, double sigma, double lower,
                                  double upper, std::mt19937_64 &rng) {
  double u =
      stan::math::uniform_rng(stan::math::Phi((lower - mu) / sigma),
                              stan::math::Phi((upper - mu) / sigma), rng);

  double tmp = stan::math::inv_Phi(u);
  return sigma * tmp + mu;
}

double trunc_normal_lpdf(double x, double mu, double sigma, double lower,
                         double upper) {
  if ((x < lower) || (x > upper)) return stan::math::NEGATIVE_INFTY;

  double out = stan::math::normal_lpdf(x, mu, sigma);
  out -= stan::math::log_diff_exp(stan::math::normal_lcdf(upper, mu, sigma),
                                  stan::math::normal_lcdf(lower, mu, sigma));

  return out;
}

void to_proto(const MatrixXd &mat, EigenMatrix *out) {
  out->set_rows(mat.rows());
  out->set_cols(mat.cols());
  *out->mutable_data() = {mat.data(), mat.data() + mat.size()};
}

void to_proto(const VectorXd &vec, EigenVector *out) {
  out->set_size(vec.size());
  *out->mutable_data() = {vec.data(), vec.data() + vec.size()};
}

VectorXd to_eigen(const EigenVector &vec) {
  int size = vec.size();
  Eigen::VectorXd out;
  if (size > 0) {
    const double *p = &(vec.data())[0];
    out = Map<const VectorXd>(p, size);
  }
  return out;
}

MatrixXd to_eigen(const EigenMatrix &mat) {
  int nrow = mat.rows();
  int ncol = mat.cols();
  Eigen::MatrixXd out;
  if (nrow > 0 & ncol > 0) {
    const double *p = &(mat.data())[0];
    out = Map<const MatrixXd>(p, nrow, ncol);
  }
  return out;
}

std::vector<VectorXd> to_vector_of_vectors(const MatrixXd &mat) {
  std::vector<VectorXd> out(mat.rows());
  for (int i = 0; i < mat.rows(); i++) out[i] = mat.row(i).transpose();

  return out;
}

// element i,j of D contains the squaredNorm of the difference of row i of x and row j of y
// if matrices contain points, element i,j of D is the squared distance between point i of
// x and point j of y
MatrixXd pairwise_dist_sq(const MatrixXd &x, const MatrixXd &y) {
  MatrixXd D(x.rows(), y.rows());
  int i = 0;
  for (int i = 0; i < y.rows(); i++)
    D.col(i) = (x.rowwise() - y.row(i)).rowwise().squaredNorm();

  return D;
}

// if x contains points on different lines, the returned D has element i,j
// equal to the distance between point i and point j (squared)
MatrixXd pairwise_dist_sq(const MatrixXd &x) { return pairwise_dist_sq(x, x); }


VectorXd softmax(const VectorXd &logs) {
  VectorXd num = (logs.array() - logs.maxCoeff()).exp();
  return num / num.sum();
  // after a "strange" step, what it returns is (p1/P,...,pM/P), where P=sum(pi)
  // what it receives is (ln(p1),...,ln(pM))
}

MatrixXd posterior_sim_matrix(const MatrixXi &alloc_chain) {
  MatrixXd out(alloc_chain.cols(), alloc_chain.cols());
  for (int i = 1; i < alloc_chain.cols(); i++) {
    for (int j = 1; j < alloc_chain.cols(); j++) {
      double val = (alloc_chain.col(i).array() == alloc_chain.col(j).array())
                       .cast<double>()
                       .mean();
      out(i, j) = val;
      out(j, i) = val;
    }
  }
  return out;
}

// VectorXd minbinder(const MatrixXi &alloc_chain) {
//   MatrixXd psm = posposterior_sim_matrix(alloc_chain);
//   std::vector<double> dists;

//   std::transform(
//     alloc_chain.rowwise().begin(), alloc_chain.rowwise().end(),
//     std::back_inserter(dists),
//     [&psm](const VectorXi &clus){
//       int maxclus = clus.maxCoeff();
//       MatrixXd coclust = MatrixXd::Zero(clus.size(), clus)
//       for (int k=0; k < maxclus; k++) {

//       }
//      })
// }
