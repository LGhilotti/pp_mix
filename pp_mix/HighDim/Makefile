ROOT_DIR := .
SRC_DIR := $(ROOT_DIR)/src
SPIKES_DIR := $(SRC_DIR)/spikes
STAN_ROOT_DIR := $(if $(STAN_ROOT_DIR), $(STAN_ROOT_DIR), /home/mario/dev/bayesmix/lib/math)

CXX = g++
CFLAGS = \
	-std=c++1y \
	-MMD \
	-I$(STAN_ROOT_DIR) \
	-I$(ROOT_DIR)/eigen/ \
	-I$(STAN_ROOT_DIR)/lib/boost_*/  \
	-I$(STAN_ROOT_DIR)/lib/sundials_*/include  \
	-I$(STAN_ROOT_DIR)/lib/tbb_2019_U8/include  \
	-D_REENTRANT -fPIC \
	-O3 -march=native -msse2 -funroll-loops -ftree-vectorize -fopenmp
LDLIBS = \
 	-L$(STAN_ROOT_DIR)/lib/tbb \
	-lgsl -lgslcblas -lpthread -ltbb -Wl,-rpath,"$(STAN_ROOT_DIR)/lib/tbb"
LDFLAGS = -D_REENTRANT -O3 -fopenmp

SPIKES_SRCS = $(SPIKES_DIR)/simulate_gig.cpp
#OUR_SRCS = $(SRC_DIR)/rng.cpp

#OBJS = $(subst .cpp,.o, $(OUR_SRCS))

SPIKES_EXECS = $(subst .cpp,.out, $(SPIKES_SRCS))
SPIKES_OBJS =  $(subst .cpp,.o, $(SPIKES_SRCS))

test: $(SPIKES_EXECS)

$(SPIKES_EXECS): %.out: %.o #$(OBJS)
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJS) $< $(LDLIBS)

%.o : %.cpp
	$(CXX) $(CFLAGS) -MMD -MP -c $< -o $@

clean:
	rm $(SPIKES_OBJS) #$(OBJS)

distclean: clean
