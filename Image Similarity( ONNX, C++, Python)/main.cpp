#include <iostream>
#include "torch/script.h"
#include "torch/torch.h"
#include <eigen3/Eigen/Dense>

// torch::Tensor X = torch::zeros({1,3,224,224});
// torch::Tensor custom_group_norm()
using ConstEigenVectorArrayMap = Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, 1>>;
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>>;
// std::out << "*******************************" << std::endl;
torch::Tensor custom_group_norm(torch::Tensor X)
 {
    // torch::Tensor X = torch::zeros({1,3,224,224});
    // std::cout << "\n\n\n\n\n\nstd started................88888888.....................................==========\n\n\n\n" << std::endl;
    torch::jit::script::Module module ;
    module = torch::jit::load("/home/mitesh/files_backup/onnx_proj/bhavai/models/script_module_2.pt");
    std::vector<torch::jit::IValue> input;
    input.push_back(X);
    torch::Tensor out = module(input).toTensor().squeeze();
    return  out;

    // std::cout << out << std::endl;
}



// int main()
// {
//   std::cout << "hi";
//   return 0;
// }


static auto registry =
  torch::RegisterOperators("mynamespace::custom_group_norm", &custom_group_norm);
