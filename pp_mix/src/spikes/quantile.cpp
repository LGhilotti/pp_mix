#include <boost/math/distributions/chi_squared.hpp>
#include <iostream>

int main() {
    boost::math::chi_squared chisq(3);
    std::cout << quantile(complement(chisq, 0.1)) << std::endl;
    std::cout << quantile(complement(chisq, 0.9)) << std::endl;
}