#ifndef BASE_JUMP_HPP
#define BASE_JUMP_HPP

#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

class BaseJump {
 public:
    virtual ~BaseJump() {};

    // Sample from full-conditional of s^(na) : non allocated components
    virtual double sample_tilted(double u) = 0;

    // Sample from full-conditional of s^(a): allocated components
    virtual double sample_given_data(
        int ndata, double curr, double u) = 0;

    virtual double laplace(double u) = 0;
};


#endif
