#include "base_pp.hpp"

BasePP::BasePP(const MatrixXd &ranges): ranges(ranges) {
    dim = ranges.cols();
    diff_range = (ranges.row(1) - ranges.row(0)).transpose();
    vol_range = diff_range.prod();
}

void BasePP::set_ranges(const MatrixXd &ranges) {
    this->ranges = ranges;
    dim = ranges.cols();
    // std::cout << "ranges: \n" << ranges << std::endl;
    diff_range = (ranges.row(1) - ranges.row(0)).transpose();
    vol_range = diff_range.prod();

    initialize();
}

MatrixXd BasePP::sample_uniform(int npoints)
{
    MatrixXd out(npoints, dim);
    for (int j=0; j < dim; j++) {
        for (int i=0; i < npoints; i++) {
            out(i, j) = uniform_rng(
                ranges(0, j), ranges(1, j), Rng::Instance().get());
        }
    }    

    return out;
}

void BasePP::sample_given_active(
        const MatrixXd &active, MatrixXd *non_active, double psi_u) {
    int npoints = non_active->rows();
    double c_star_na = c_star * psi_u;

    std::cout << "**********************" << std::endl;
    std::cout << "c_star: " << c_star << std::endl;
    std::cout << "npoints: " << npoints << std::endl;

    birth_prob = std::log(c_star) - std::log(c_star + npoints);
    std::cout << "birth_prob: " << birth_prob << std::endl;
    double rsecond = uniform_rng(0, 1, Rng::Instance().get());

    birth_arate = -1;

    if (std::log(rsecond) < birth_prob) {
        // BIRTH MOVE
        // std::cout << "proposing birth" << std::endl;
        VectorXd xi = phi_star_rng();
        MatrixXd aux(active.rows() + npoints, dim);
        aux << active, *non_active;
        double pap = papangelou(xi, aux);
        birth_arate = pap - phi_star_dens(xi) + std::log(psi_u);
        
        // std::cout << "xi: " << xi.transpose() << std::endl;;
        // std::cout << "active: \n" << active << std::endl;
        // std::cout << "non active \n" << *non_active << std::endl;
        // std::cout << "aux: \n" << aux << std::endl;
        // std::cout << "papangelou: " << pap
        //           << ", phi_star_dens: " << phi_star_dens(xi)
        //           << ", log_psi: " << std::log(psi_u) << std::endl;

        double rthird = uniform_rng(0, 1, Rng::Instance().get());
        if (std::log(rthird) < birth_arate)
        {
            // std::cout << "accepted birth" << std::endl;
            // std::cout << "pap: " << pap << std::endl;
            // assert(pap > -3.0);
            non_active->conservativeResize(npoints + 1, dim);
            non_active->row(npoints) = xi;
        }
    } else {
        // Death Move
        if (npoints == 0)
            return;

        VectorXd probas = VectorXd::Ones(npoints) / npoints;
        int ind = categorical_rng(probas, Rng::Instance().get()) - 1;

        delete_row(non_active, ind);
    }
    // std::cout << "**********************" << std::endl;

    return;
}