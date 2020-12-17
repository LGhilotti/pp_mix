#include "../gig.hpp"
#include <numeric>
using namespace Eigen;

int main(){
  int n=10000;
  ArrayXd samples(n);
  std::vector<double> res(GIG::rgig(n,-3,2,0));
  for (int i=0;i<n;i++){
    samples(i)=res[i];
  }
  std::cout<< "Mean: "<<samples.mean()<<std::endl;
  std::cout<<"Variance: "<<(samples - samples.mean()).square().sum()/(samples.size()-1)<<std::endl;
  std::cout <<"sampled value:"<< std::endl;
  for (int i=0;i<100;i++){
    std::cout<<samples[i]<<std::endl;
  }

  return 0;

}
