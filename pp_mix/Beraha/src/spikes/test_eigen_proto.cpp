#include <Eigen/Dense>
#include "../../protos/cpp/state.pb.h"

using namespace Eigen;

void _to_proto(const VectorXd &vec, EigenVector *out)
{
    out->set_size(vec.size());
    *out->mutable_data() = {vec.data(), vec.data() + vec.size()};
}

Eigen::VectorXd _to_eigen(const EigenVector& vec)
{
    int size = vec.size();
    const double *p = &(vec.data())[0];
    Map<const VectorXd> out(p, size);
    return out;
}

void _to_proto(const MatrixXd &mat, EigenMatrix *out)
{
    out->set_rows(mat.rows());
    out->set_cols(mat.cols());
    *out->mutable_data() = {mat.data(), mat.data() + mat.size()};
}

MatrixXd _to_eigen(const EigenMatrix &mat)
{
    int nrow = mat.rows();
    int ncol = mat.cols();
    const double *p = &(mat.data())[0];
    Map<const MatrixXd> out(p, nrow, ncol);
    return out;
}

int main() {
    MatrixXd a(2, 2);
    a << 1, 2, 3, 4;
    std::cout << "a: \n" << a << std::endl;

    EigenMatrix aproto;
    _to_proto(a, &aproto);
    aproto.PrintDebugString();

    MatrixXd avector = _to_eigen(aproto);
    std::cout << "avector: \n" << avector << std::endl;;
}