cmake_minimum_required(VERSION 3.5.1)

project(rsmproject VERSION 1.0.0)

set(PROJECT_SOURCE_DIR /home/wonki/rsm-simul/)
set(SRCS 
    leveldb.cpp
    util.cpp
    policy_rl/NDQNTrainer.cc
    policy_rl/DDPGTrainer.cc
    policy_rl/DQNTrainer.cc
    policy_rl/ExperienceReplay.cc)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

#add_library(util util.h util.cpp)
#add_library(leveldb leveldb.h leveldb.cpp)
add_library(leveldb ${SRCS})

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/policy_rl)

set(Torch_DIR ${PROJECT_SOURCE_DIR}/libtorch/share/cmake/Torch)
find_package(Torch CONFIG REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${TORCH_INCLUDE_DIRS})
find_package(GSL REQUIRED)
include_directories(/usr/local/include/gsl)

add_executable(main main.cpp)
target_link_libraries(main leveldb ${TORCH_LIBRARIES} ${GSL_LIBRARIES})
