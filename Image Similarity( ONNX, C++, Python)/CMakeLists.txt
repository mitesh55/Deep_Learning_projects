cmake_minimum_required(VERSION 3.1)
project(heet)
set(CMAKE_PREFIX_PATH /home/mitesh/Downloads/libtorch-cxx11-abi-shared-with-deps-1.9.1+cpu/libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}" )
add_executable(heet main.cpp)
target_link_libraries(heet "${TORCH_LIBRARIES}")
set_property(TARGET heet PROPERTY CXX_STANDARD 14)

	
