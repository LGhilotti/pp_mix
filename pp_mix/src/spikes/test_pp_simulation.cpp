#include "../point_process/strauss_pp.hpp"
#include "../simulate_straus.hpp"
#include "../point_process/perfect_sampler.hpp"


int main() {
    StraussPP pp2(1.0, 0.1, 1.0);

    BasePP *pp;
    pp = new StraussPP(0.01, 0.01, 3.0);
    Eigen::MatrixXd ranges(2, 2);
    ranges << 0.0, 0.0, 10.0, 10.0;
    pp->set_ranges(ranges);

    // PerfectSampler sampler(pp);

    // std::cout << sampler.simulate() << std::endl;

    std::cout << pp->sample_n_points(5) << std::endl;

    // std::cout << simulate_strauss_our(&pp) << std::endl;

    // std::cout << simulate_strauss_moller(ranges, 0.08, 0.9, 0.3) << std::endl;
}