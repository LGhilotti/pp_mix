#include "perfect_sampler.hpp"

void PerfectSampler::initialize() {
  in_upper.resize(0);
  in_lower.resize(0);
  // initialize points with the dominating poisson process
  int npoints = stan::math::poisson_rng(pp->get_cstar(), Rng::Instance().get());
  for (int i = 0; i < npoints; i++) {
    Point p;
    p.coords = pp->phi_star_rng();
    p.number = max_id;
    max_id += 1;
    state.push_front(p);
    id2point[p.number] = p;
  }
  std::cout << "init npoints: " << npoints << std::endl;
}

void PerfectSampler::estimate_doubling_time() {
  // run the perfect bacwkard birth and death untill all the original points
  // have been deleted
  int start_npoints = state.size();
  int t = 0;
  while (start_npoints > 0) {
    one_backward(&state);
    if (std::get<1>(transitions.front()) &
        std::get<0>(transitions.front()) <= start_npoints) {
      start_npoints -= 1;
    }
    t++;
  }
  double_t = t;
  transitions.clear();
}

void PerfectSampler::one_backward(std::deque<Point>* points) {
  double rsec = stan::math::uniform_rng(0, 1, Rng::Instance().get());
  double cstar = pp->get_cstar();
  if (rsec > cstar / (cstar + points->size())) {
    if (points->size() == 0) return;

    double rthird = stan::math::uniform_rng(0, 1, Rng::Instance().get());
    VectorXd probas = VectorXd::Ones(points->size()) / points->size();
    int removed = categorical_rng(probas, Rng::Instance().get()) - 1;
    Point p = *(points->begin() + removed);
    p.r_mark = rthird;
    points->erase(points->begin() + removed);
    // deaths.push_front(p);
    transitions.push_front(std::make_tuple(p.number, true, rthird));
  } else {
    Point p;
    p.coords = pp->phi_star_rng();
    p.number = max_id;
    points->push_front(p);
    max_id += 1;
    transitions.push_front(std::make_tuple(p.number, false, -1.0));
    id2point[p.number] = p;
  }
}

void PerfectSampler::one_forward(std::tuple<int, bool, double> trans) {
  Point curr = id2point[std::get<0>(trans)];
  if (std::get<1>(trans)) {
    std::list<Point> upper;
    std::list<Point> lower;
    for (const int& id : in_upper) {
      upper.push_front(id2point[id]);
    }

    for (const int& id : in_lower) {
      lower.push_front(id2point[id]);
    }

    double rthird = std::get<2>(trans);
    if (std::log(rthird) <
        pp->papangelou(curr, lower) - pp->phi_star_dens(curr.coords)) {
      in_upper.push_front(curr.number);
      numupper++;
    }
    if (std::log(rthird) <
        pp->papangelou(curr, upper) - pp->phi_star_dens(curr.coords)) {
      in_lower.push_front(curr.number);
      numlower++;
    }

    // state.push_back(curr);

  } else {
    if (in_upper.size() > 0) {
      in_upper.erase(
          std::remove_if(in_upper.begin(), in_upper.end(),
                         [&curr](int id) { return id == curr.number; }));
    }
    if (in_lower.size() > 0) {
      auto it = std::remove_if(in_lower.begin(), in_lower.end(),
                               [&curr](int id) { return id == curr.number; });
      in_lower.erase(it);
    }
  }
}

MatrixXd PerfectSampler::simulate() {
  initialize();

  estimate_doubling_time();
  if (state.size() == 0) {
    MatrixXd out(state.size(), pp->get_dim());
    for (int i = 0; i < state.size(); i++) {
      out.row(i) = state[i].coords.transpose();
    }
    return out;
  }

  int start_t = 0;
  int end_t = double_t;
  bool is_first = true;
  bool coalesced = false;

  while (!coalesced) {
    for (int i = start_t; i < end_t; i++) {
      one_backward(&state);
    }

    in_upper.clear();
    for (const Point& p : state) in_upper.push_front(p.number);

    in_lower.clear();
    for (int i = 0; i < end_t; i++) {
      auto trans_it = transitions.begin() + i;
      one_forward(*trans_it);
    }

    start_t = end_t;
    end_t *= 2;
    coalesced = (in_lower.size() == in_upper.size());
  }

  MatrixXd out(in_upper.size(), pp->get_dim());
  for (int i = 0; i < in_upper.size(); i++) {
    int id = *(in_upper.begin() + i);
    out.row(i) = id2point[id].coords.transpose();
  }
  return out;
}
