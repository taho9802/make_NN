#include <iostream>
#include <torch/torch.h>
#include "neural_arch.h"

int main(){
  NetImpl network(50, 10);
  std::cout << network << "\n\n";
}