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
    birth_prob = std::log(c_star) - std::log(c_star + npoints);

    // std::cout << "**********************" << std::endl;
    // std::cout << "c_star: " << c_star << std::endl;
    // std::cout << "npoints: " << npoints << std::endl;
    // std::cout << "birth_prob: " << std::exp(birth_prob) << std::endl;
    double rsecond = uniform_rng(0, 1, Rng::Instance().get());

    birth_arate = -1;

    if (std::log(rsecond) < birth_prob) {
        // BIRTH MOVE
        // std::cout << "proposing birth" << std::endl;
        VectorXd xi = phi_star_rng();
        MatrixXd aux(active.rows() + npoints, dim);
        aux << active, *non_active;
        double pap = papangelou(xi, aux);
        birth_arate = pap + std::log(psi_u); // - phi_star_dens(xi); // + std::log(psi_u);
        // std::cout << "birth_arate: " << std::exp(birth_arate) << std::endl;
        // std::cout << "pap: " << pap << ", phi_dens: "
        //           << phi_star_dens(xi) << std::endl;

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

        // std::cout << "death move" << std::endl;
        VectorXd probas = VectorXd::Ones(npoints) / npoints;
        int ind = categorical_rng(probas, Rng::Instance().get()) - 1;

        delete_row(non_active, ind);
    }
    // std::cout << "**********************" << std::endl;

    return;
}

MatrixXd BasePP::sample_n_points(int npoints) {
    int max_steps = 1e6;
    MatrixXd out(npoints, dim);
    double logM = std::log(rejection_sampling_M(npoints));
    std::cout << "logM: " << logM << std::endl;

    for (int i=0; i < max_steps; i++) {
        double dens_q = 0;
        for (int k=0; k < npoints; k++) {
            out.row(k) = phi_star_rng().transpose();
            dens_q += phi_star_dens(out.row(k).transpose(), true);
        }

        std::cout << "dens(out): " << dens(out) << std::endl;
        std::cout << "dens_q: " << dens_q << std::endl;
        double arate = dens(out) - (logM + dens_q);
        double u = stan::math::uniform_rng(0.0, 1.0, Rng::Instance().get());
        std::cout << "arate: " << arate << std::endl;
        std::cout << "u: " << std::log(u) << std::endl;


        if (std::log(u) < arate)
            return out;
    }

    std::cout << "MAXIMUM NUMBER OF ITERATIONS REACHED IN "
              << "BasePP::sample_n_points, returning the last value" << std::endl;

    return out;
}
