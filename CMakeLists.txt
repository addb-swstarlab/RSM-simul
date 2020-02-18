cmake_minimum_required(VERSION 3.5.1)

project(rsmproject VERSION 1.0.0)

set(PROJECT_SOURCE_DIR /home/wonki/rsm-simul/)

add_library(util util.h util.cpp)
add_library(leveldb leveldb.h leveldb.cpp)

include_directories(${PROJECT_SOURCE_DIR}/policy_rl)

set(Torch_DIR ${PROJECT_SOURCE_DIR}/libtorch/share/cmake/Torch)
find_package(Torch CONFIG REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
find_package(GSL REQUIRED)
include_directories(/usr/local/include/gsl)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE util leveldb)

