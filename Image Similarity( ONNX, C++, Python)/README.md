# Image Similarity 

## Project Agenda

In current era, Technology is growing enormously such as **Self Driving Car**.Which makes calculation of its surroundings within fraction of second to make decision withing fraction of seconds, At that time latency of any system is major concern. 

This project Aims on reducing latency by making custom operator in C++ language due to its memory management system.

### Tools used : ONNX, Python & C++

**ONNX** is a common file format that enables AI developers to use models with a variety of frameworks, tools, runtimes, and compilers.

### Steps Included for ONNX Implementation

- [1](#step1) - Adding the custom operator implementation in C++ and registering it with TorchScript
  - [2](#step2) - Exporting the custom Operator to ONNX, using:
  <br />             - a combination of existing ONNX ops
  <br />              or
  <br />              - a custom ONNX Operator

**C++** is been used to make custom operator to imerge latency in calculation due to it's memory managment system.
* To Register Custom Operator

```cpp
static auto registry = torch::RegisterOperators("mynamespace::custom_group_norm", &custom_group_norm);
```
go to instructions.txt for more information.

Reference : [https://github.com/onnx/tutorials/edit/master/PyTorchCustomOperator/](https://github.com/onnx/tutorials/edit/master/PyTorchCustomOperator/)

