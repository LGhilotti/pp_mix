#ifndef FACTORY_HPP
#define FACTORY_HPP

#include <memory>

#include "covs/base_cov.hpp"
#include "covs/fixed_cov.hpp"
#include "covs/inv_wishart.hpp"

#include "jumps/base_jump.hpp"
#include "jumps/gamma.hpp"

#include "point_process/base_pp.hpp"
#include "point_process/strauss_pp.hpp"

#include "conditional_mcmc.hpp"

#include "../protos/cpp/params.pb.h"

BasePP* make_pp(const Params& params);


BasePP* make_strauss(const StraussParams& params);


BaseJump* make_jump(const Params& params);

BaseJump* make_gamma_jump(const GammaParams& params);


BasePrec* make_prec(const Params& params);

BasePrec* make_fixed_prec(const FixedPrecParams& params);

BasePrec* make_wishart(const WishartParams& params);

#endif