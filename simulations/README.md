This repository contains code for the paper "MCMC computations for Bayesian mixture models using repulsive point processes" by Mario Beraha, Raffaele Argiento, Jesper M{\o}ller and Alessandra Guglielmi (soon on arXiv)

This is not a general-purpose statistical software, use at your own risk!

The code consists of a C++ library that computes MCMC posterior simulations for the models described in the paper, and is incapsulated in a 'python-like' package, for ease of use.

## Prerequisites
1. Protobuf: First of all, install the protocol buffer library following the [instructions](https://github.com/protocolbuffers/protobuf/tree/master/src) seccondly install the python package protobuf (currently tested version 3.11.3)

2. stan/math: we make extensive use of the awesome math library developed by the Stan team. Simply clone their repo (https://github.com/stan-dev/math) in a local directory and install it.

An example of a real instantiation whenever the path to Stan Math is ~/stan-dev/math/:
```shell
make -j4 -f ~/stan-dev/math/make/standalone math-libs
make -f ~/stan-dev/math/make/standalone foo
```

Then set the environmental variable 'STAN_ROOT_DIR' to the path to 'math'.

3. pybind11
```shell
  pip3 install pybind11
```

## Installation
Installation is trivial on linux systems and has been tested only on those.

```shell
cd pp_mix
make compile_protos
make generate_pybind
```
and the package is ready to be used!

## Using the package
Just have a look at the notebooks containing the simulations from the paper to get an idea (it is pretty straightforward)
