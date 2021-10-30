from setuptools import setup, Extension
from torch.utils import cpp_extension

import torch
input = torch.rand(1,3,299,299)
# print("error!")
def register_custom_op():
    def my_group_norm(g, input):
        return g.op("mydomain::testgroupnorm", input)

    from torch.onnx import register_custom_op_symbolic
    # print("***********************************************************")
    # # my_group_norm(input=input)  
    # print("==============================================")                                                                      #
    register_custom_op_symbolic("mynamespace::custom_group_norm", my_group_norm, 9)


def export_custom_op():
    class CustomModel(torch.nn.Module):   # torch.nn.Module
        def forward(self, x):
            return torch.ops.mynamespace.custom_group_norm(x)

# def export_custom_op():
#     def CustomModel(x):
#         return torch.ops.mynamespace.custom_group_norm(x)
    inputs = torch.tensor(torch.zeros(1,3,299,299).detach())

    f = './model.onnx'
    torch.onnx.export(CustomModel(), inputs, f,
                      opset_version=9,
                      example_outputs=None,
                      input_names=["X"], output_names=["Y"],
                      custom_opsets={"mydomain": 1})
path = "/home/mitesh/files_backup/onnx_proj/bhavai/build/lib.linux-x86_64-3.8/custom_group_norm.cpython-38-x86_64-linux-gnu.so"
#"build/lib.linux-x86_64-3.7/custom_group_norm.cpython-37m-x86_64-linux-gnu.so"
torch.ops.load_library(path)
register_custom_op()
export_custom_op()
out_1= torch.ops.mynamespace.custom_group_norm(input)
#out_2 = torch.ops.mynamespace.custom_group_norm(torch.ones(1,3,224,224))
#print(out_1, out_2)
print(out_1)

