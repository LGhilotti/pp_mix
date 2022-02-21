#include "determinantalPP.hpp"

#include <numeric>

using stan::math::LOG_SQRT_PI;
double PI = stan::math::pi();


DeterminantalPP::DeterminantalPP(const MatrixXd &ranges, int N, double c, double s):
  ranges(ranges), N(N), c(c), s(s) {

  dim = ranges.cols();
  diff_range = (ranges.row(1) - ranges.row(0)).transpose();
  vol_range = diff_range.prod();

  A = MatrixXd::Zero(dim, dim);
  b = VectorXd::Zero(dim);
  for (int i = 0; i < dim; i++) {
    A(i, i) = 1.0 / (ranges(1, i) - ranges(0, i));
    b(i) = -A(i, i) * (ranges(1, i) + ranges(0, i)) / 2.0;
  }

  compute_Kappas();
  Kappas_red.resize(Kappas.rows()/2 +1,dim);
  Kappas_red = Kappas.bottomRows(Kappas.rows()/2 +1);

  phis.resize(Kappas.rows());
  phi_tildes.resize(Kappas.rows());
  phis_tmp.resize(Kappas.rows());
  phi_tildes_tmp.resize(Kappas.rows());

  phis_red.resize(Kappas_red.rows());
  phi_tildes_red.resize(Kappas_red.rows());
  phis_tmp_red.resize(Kappas_red.rows());
  phi_tildes_tmp_red.resize(Kappas_red.rows());

  /*
  std::cout << "ranges: "<<this->ranges<<std::endl;
  std::cout << "N: "<<this->N<<std::endl;
  std::cout << "c: "<<this->c<<std::endl;
  std::cout << "s: "<<this->s<<std::endl;
  std::cout << "dim: "<<this->dim<<std::endl;
  std::cout << "diff_range: "<<this->diff_range<<std::endl;
  std::cout << "vol_range: "<<this->vol_range<<std::endl;
  std::cout<<" end Base constructor"<<std::endl;
  */
  return;
}


void DeterminantalPP::set_decomposition(const MatrixXd * lambda) {

  Lambda = lambda;
  compute_eigen_and_cstar(&Ds, &phis, &phi_tildes, &c_star, lambda);
  //compute_eigen_and_cstar_red(&Ds_red, &phis_red, &phi_tildes_red, &c_star_red, lambda);
  return;

}


void DeterminantalPP::compute_eigen_and_cstar(double * D_, VectorXd * Phis_, VectorXd * Phi_tildes_, double * C_star_, const MatrixXd * lambda){


  *D_ = 0.0;
  *C_star_ = 0.0;

  LLT<MatrixXd> M ((*lambda).transpose() * (*lambda));
  // compute determinant of Lambda^T Lambda
  double det = std::pow(M.matrixL().determinant(),2);

  double esp_fact = -2*std::pow(stan::math::pi(),2)*std::pow(det,1.0/dim)*std::pow(c,-2.0/dim);
  ArrayXd vec_phi(Kappas.rows());
  //#pragma omp parallel for default(none) shared(vec_phi, Kappas, s,esp_fact,M)
  for (int i = 0; i < Kappas.rows(); i++) {
    VectorXd sol = M.solve(Kappas.row(i).transpose());
    double dot_prod = (Kappas.row(i)).dot(sol);
    vec_phi(i) = s*std::exp(esp_fact*dot_prod);
  }

  *Phis_ = vec_phi.matrix();
  *Phi_tildes_ = (vec_phi / (1 - vec_phi)).matrix();

  *D_ = log(1 + (*Phi_tildes_).array()).sum();
  *C_star_ = (*Phi_tildes_).sum();

  return;

}
/*
void DeterminantalPP::compute_eigen_and_cstar_red(double * D_, VectorXd * Phis_, VectorXd * Phi_tildes_, double * C_star_, const MatrixXd * lambda){


  *D_ = 0.0;
  *C_star_ = 0.0;

  LLT<MatrixXd> M ((*lambda).transpose() * (*lambda));
  // compute determinant of Lambda^T Lambda
  double det = std::pow(M.matrixL().determinant(),2);

  double esp_fact = -2*std::pow(stan::math::pi(),2)*std::pow(det,1.0/dim)*std::pow(c,-2.0/dim);
//  ArrayXd vec_phi(Kappas.rows());
  //#pragma omp parallel for default(none) shared(vec_phi, Kappas, s,esp_fact,M)
  for (int i = 0; i < Kappas_red.rows(); i++) {
    VectorXd sol = M.solve(Kappas_red.row(i).transpose());
    double dot_prod = (Kappas_red.row(i)).dot(sol);
    (*Phis_)(i) = s*std::exp(esp_fact*dot_prod);
  }

  //*Phis_ = vec_phi.matrix();
  *Phi_tildes_ = ((*Phis_).array() / (1 - (*Phis_).array())).matrix();

  *D_ = log(1 + (*Phi_tildes_).array()).sum() * 2.0 - log(1+(*Phi_tildes_)(0));
  *C_star_ = (*Phi_tildes_).sum() * 2.0 - (*Phi_tildes_)(0);

  return;

}
*/

void DeterminantalPP::decompose_proposal(const MatrixXd& lambda) {

  compute_eigen_and_cstar(&Ds_tmp, &phis_tmp, &phi_tildes_tmp, &c_star_tmp, &lambda);
  //compute_eigen_and_cstar_red(&Ds_tmp_red, &phis_tmp_red, &phi_tildes_tmp_red, &c_star_tmp_red, &lambda);

  return;

}


void DeterminantalPP::update_decomposition_from_proposal() {

  std::swap(Ds, Ds_tmp);
  phis.swap(phis_tmp);
  phi_tildes.swap(phi_tildes_tmp);
  std::swap(c_star, c_star_tmp);
/*
  std::swap(Ds_red, Ds_tmp_red);
  phis_red.swap(phis_tmp_red);
  phi_tildes_red.swap(phi_tildes_tmp_red);
  std::swap(c_star_red, c_star_tmp_red);*/
  return;
}

// compute just once the grid for summation over Z^dim
void DeterminantalPP::compute_Kappas() {

  std::vector<double> k(2 * N + 1);
  for (int n = -N; n <= N; n++) {
    k[n + N] = n;
  }
  std::vector<std::vector<double>> kappas;
  if (dim == 1) {
    kappas.resize(k.size());
    for (int i = 0; i < k.size(); i++) kappas[i].push_back(k[i]);
  } else {
    kappas = cart_product(k, dim);
  }

  Kappas.resize(kappas.size(), dim);
  for (int i = 0; i < kappas.size(); i++) {
    Kappas.row(i) = Map<VectorXd>(kappas[i].data(), dim).transpose();
  }


  return;

}


double DeterminantalPP::dens_cond_in_proposal(const MatrixXd& x, bool log) {

  double out = ln_dens_process(x, Ds_tmp, phis_tmp, phi_tildes_tmp, c_star_tmp);
  out -= std::log(1-std::exp(-Ds_tmp));

  if (!log) out=std::exp(out);

  return out;

}


double DeterminantalPP::dens_cond(const MatrixXd& x, bool log) {

  double out = ln_dens_process(x, Ds, phis, phi_tildes, c_star);
  out -= std::log(1-std::exp(-Ds));

  if (!log) out = std::exp(out);

  return out;

}
/*
//test reduced
double DeterminantalPP::dens_cond_red(const MatrixXd& x, bool log) {

  double out = ln_dens_process_red(x, Ds_red, phis_red, phi_tildes_red, c_star_red);
  out -= std::log(1-std::exp(-Ds_red));

  if (!log) out = std::exp(out);

  return out;

}

double DeterminantalPP::ln_dens_process_red(const MatrixXd& x, double Ds_p, const VectorXd& phis_p,
            const VectorXd& phi_tildes_p, double c_star_p){

  double out;
  int n;

  // check if it's jut one point
  if ((x.size() == 1 && dim == 1) || (x.rows() == 1 & dim > 1) ||
      (x.cols() == 1 && dim > 1)) {
    n = 1;
    out =
        -1.0 * n * std::log(vol_range) - Ds_p + std::log(c_star_p);
  }
  else {
    int n = x.rows();
    bool check_range = true;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < dim; j++) {
        if (x(i, j) < ranges(0, j) || x(i, j) > ranges(1, j))
          check_range = false;
      }
    }
    if (check_range) {
      out = -1.0 * n * std::log(vol_range) - Ds_p;


      // Transform data points to be in the unit cube centered in 0
      MatrixXd xtrans(n,x.cols());

      for (int i = 0; i < n; i++)
        xtrans.row(i) = (A * x.row(i).transpose() + b).transpose();

      // std::cout << "xtrans " << xtrans.transpose() << std::endl;
      out += log_det_Ctilde_red(xtrans, phi_tildes_p);

    } else {
      out = stan::math::NEGATIVE_INFTY;
    }
  }
  // std::cout << "dens: " << out << std::endl;

  return out;

}

double DeterminantalPP::log_det_Ctilde_red(const MatrixXd &x, const VectorXd& phi_tildes_p) {
  MatrixXd Ctilde(x.rows(), x.rows());

  for (int l = 0; l < x.rows()-1; l++) {
    for (int m = l+1; m < x.rows(); m++) {
      double aux = 0.0;
      RowVectorXd vec(x.row(l)-x.row(m));
      //int nthreads;
      //#pragma omp parallel for default(none) firstprivate(Kappas,vec, phi_tildes_p) reduction(+:aux)
      for (int kind = 1; kind < Kappas_red.rows(); kind++) {
        //nthreads = omp_get_num_threads();
        //printf("Number of threads = %d\n", nthreads);
        double dotprod = Kappas_red.row(kind).dot(vec);
        aux += phi_tildes_p[kind] * std::cos(2. * stan::math::pi() * dotprod);
      }
      Ctilde(l, m) = 2.0*aux + phi_tildes_p(0) ;
      if (l!=m) Ctilde(m,l) = 2.0*aux + phi_tildes_p(0);
    }
  }
  //Ctilde.diagonal() = ArrayXd::Constant(x.rows(), 2.*phi_tildes_p.sum() - phi_tildes_p(0));
  Ctilde.diagonal() = ArrayXd::Constant(x.rows(), 2.*phi_tildes_p.tail(phi_tildes_p.rows()-1).sum() + phi_tildes_p(0));

  std::cout<<"Ctilde_red:\n"<<Ctilde<<std::endl;
  std::cout<<"log_det_Ctilde_red:\n"<<2.0 * std::log(Ctilde.llt().matrixL().determinant())<<std::endl;

  return 2.0 * std::log(Ctilde.llt().matrixL().determinant());
}
//end test reduced

double DeterminantalPP::dens(const MatrixXd &x, bool log) {

  double out = ln_dens_process(x, Ds, phis, phi_tildes, c_star);

  if (!log) out = std::exp(out);

  return out;

}
*/

double DeterminantalPP::ln_dens_process(const MatrixXd& x, double Ds_p, const VectorXd& phis_p,
            const VectorXd& phi_tildes_p, double c_star_p){

  double out;
  int n;

  // check if it's jut one point
  if ((x.size() == 1 && dim == 1) || (x.rows() == 1 & dim > 1) ||
      (x.cols() == 1 && dim > 1)) {
    n = 1;
    out =
        -1.0 * n * std::log(vol_range) - Ds_p + std::log(c_star_p);
  }
  else {
    int n = x.rows();
    bool check_range = true;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < dim; j++) {
        if (x(i, j) < ranges(0, j) || x(i, j) > ranges(1, j))
          check_range = false;
      }
    }
    if (check_range) {
      out = -1.0 * n * std::log(vol_range) - Ds_p;


      // Transform data points to be in the unit cube centered in 0
      MatrixXd xtrans(n,x.cols());

      for (int i = 0; i < n; i++)
        xtrans.row(i) = (A * x.row(i).transpose() + b).transpose();

      // std::cout << "xtrans " << xtrans.transpose() << std::endl;
      out += log_det_Ctilde(xtrans, phi_tildes_p);

    } else {
      out = stan::math::NEGATIVE_INFTY;
    }
  }
  // std::cout << "dens: " << out << std::endl;

  return out;

}


double DeterminantalPP::log_det_Ctilde(const MatrixXd &x, const VectorXd& phi_tildes_p) {
  MatrixXd Ctilde(x.rows(), x.rows());

  // TODO: Ctilde is symmetric! Also the diagonal elements are identical!
  for (int l = 0; l < x.rows(); l++) {
    for (int m = l; m < x.rows(); m++) {
      double aux = 0.0;
      RowVectorXd vec(x.row(l)-x.row(m));
      //int nthreads;
      //#pragma omp parallel for default(none) firstprivate(Kappas,vec, phi_tildes_p) reduction(+:aux)
      for (int kind = 0; kind < Kappas.rows(); kind++) {
        //nthreads = omp_get_num_threads();
        //printf("Number of threads = %d\n", nthreads);
        double dotprod = Kappas.row(kind).dot(vec);
        aux += phi_tildes_p[kind] * std::cos(2. * stan::math::pi() * dotprod);
      }
      Ctilde(l, m) = aux;
      if (l!=m) Ctilde(m,l) = aux;
    }
  }
  //std::cout<<"Ctilde:\n"<<Ctilde <<std::endl;
  return 2.0 * std::log(Ctilde.llt().matrixL().determinant());


}


double DeterminantalPP::papangelou(const VectorXd& xi, const MatrixXd &x, bool log) {
  int n = 1 + x.rows();
  MatrixXd all( n , x.cols());
  all << x, xi.transpose();
  //std::cout<<"all: "<<all<<std::endl;
  // Transform data points to be in the unit cube centered in 0
  MatrixXd alltrans(n,all.cols());
  for (int i = 0; i < n; i++)
    alltrans.row(i) = (A * all.row(i).transpose() + b).transpose();

  //std::cout<<"alltrans: "<<alltrans<<std::endl;
  double out = -1.0*std::log(vol_range)+ log_det_Ctilde(alltrans, phi_tildes) - log_det_Ctilde(alltrans.topRows(n-1), phi_tildes);
  //std::cout<<"out: "<<out<<std::endl;
  if (!log) out = std::exp(out);

  return out;
}



MatrixXd DeterminantalPP::sample_uniform(int npoints) {
  MatrixXd out(npoints, dim);
  for (int j = 0; j < dim; j++) {
    for (int i = 0; i < npoints; i++) {
      out(i, j) =
          uniform_rng(ranges(0, j), ranges(1, j), Rng::Instance().get());
    }
  }

  return out;
}


VectorXd DeterminantalPP::phi_star_rng() {
  VectorXd out(dim);
  for (int i = 0; i < dim; i++) {
    out(i) = uniform_rng(ranges(0, i), ranges(1, i), Rng::Instance().get());
  }
  return out;
}

double DeterminantalPP::phi_star_dens(VectorXd xi, bool log) {
  double out = std::log(c_star) - std::log(vol_range);
  if (!log) out = std::exp(out);

  return out;
}


void DeterminantalPP::sample_nonalloc_fullcond(MatrixXd *non_active, const MatrixXd &active,
                                 double psi_u) {
  int npoints = non_active->rows();
  double c_star_na = c_star * psi_u;
  birth_prob = std::log(c_star_na) - std::log(c_star_na + npoints);

  double rsecond = uniform_rng(0, 1, Rng::Instance().get());
  birth_arate = -1;
  if (std::log(rsecond) < birth_prob) {
    // BIRTH MOVE
    VectorXd xi = phi_star_rng();
    //std::cout<<"Birth"<<std::endl;
    // compute prob of acceptance of the new birth
    MatrixXd aux(active.rows() + npoints, dim);
  //  std::cout<<"defined aux"<<std::endl;
    aux << active, *non_active;
    //std::cout<<"filled aux: "<<aux<<std::endl;
  //  std::cout<<"rows= "<<aux.rows()<<std::endl;
  //  std::cout<<"cols= "<<aux.cols()<<std::endl;

    double pap = papangelou(xi, aux, true);
    //std::cout<<"done papan"<<std::endl;
    birth_arate = pap - phi_star_dens(xi);
    double rthird = uniform_rng(0, 1, Rng::Instance().get());
    if (std::log(rthird) < birth_arate) {
      //std::cout<<"Accepted birth"<<std::endl;
      non_active->conservativeResize(npoints + 1, dim);
      non_active->row(npoints) = xi;
    }
    //else std::cout<<"Rejected birth"<<std::endl;
  } else {
    // Death Move
    if (npoints == 0) return;
    //std::cout<<"Death"<<std::endl;
    VectorXd probas = VectorXd::Ones(npoints) / npoints;
    int ind = categorical_rng(probas, Rng::Instance().get()) - 1;

    delete_row(non_active, ind);
  }

  return;
}
