import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable


print ("models.py run...!")

resnet = models.resnet18(pretrained=True)
resnet.eval()

modules = list(resnet.children())[:-1]
resnet_model = nn.Sequential(*modules)
for p in resnet_model.parameters():
    p.requires_grad = False
resnet_model.eval()

example_input = torch.rand(1,3,299,299)
script_module = torch.jit.trace(resnet_model, example_input)
#/home/mitesh/files_backup/onnx_proj/bhavai/models

script_module.save('/home/mitesh/files_backup/onnx_proj/bhavai/models/script_module_2.pt')
# print("hi")

# my_values = {'a' : example_input}        

# class Container(torch.nn.Module):
#     def __init__(self, my_values):
#         super().__init__()
#         for key in my_values:
#             setattr(self, key, my_values[key])
# container = torch.jit.script(Container(my_values))
# container.save("container.pt")
