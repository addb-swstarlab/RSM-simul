#pragma once

#include <torch/torch.h>
#include "ExperienceReplay.h"
#include <Trainer.h>

class DQN : public torch::nn::Module {
  public:
  DQN(int64_t num_actions)
    :
    conv1(torch::nn::Conv2dOptions(1, 32, {2,1024}).stride(1)),
    conv2(torch::nn::Conv2dOptions(32, 64, {2,1}).stride(1)),

    linear1(torch::nn::Linear(64*2, 64)),
    output(torch::nn::Linear(64, num_actions)) {
      register_module("conv1", conv1);
      register_module("conv2", conv2);
      register_module("linear1", linear1);
      register_module("output", output);
  }
    
  torch::Tensor forward(torch::Tensor input) {
    input = torch::relu(conv1(input));
    input = torch::relu(conv2(input));
//        input = torch::relu(conv3(input));

  // Flatten the output
    input = input.view({input.size(0), -1});
    input = torch::relu(linear1(input));
    input = output(input);

    return input;
  }

  torch::Tensor act(torch::Tensor state){
    torch::Tensor q_value = forward(state);
    std::cout << "q_value = " << q_value << std::endl;
    torch::Tensor action = std::get<1>(q_value.max(1));

    return action;
  }

  torch::nn::Conv2d conv1, conv2;
  torch::nn::Linear linear1, output;
};


class NDQNTrainer : public Trainer {
  public:
    std::shared_ptr<DQN> dqn_local;
    std::shared_ptr<DQN> dqn_target;
    torch::optim::Adam dqn_optimizer;
    double epsilon_start = 0.9;
    double epsilon_final = 0.01;
    int64_t epsilon_decay = 30000;

    torch::Device device;
    int64_t action_size_;
        
    NDQNTrainer(int64_t action_size, int64_t capacity);
    virtual int64_t act(torch::Tensor state);
    virtual void learn();
    void hard_copy( std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target);
  
    virtual void saveCheckPoints();
    virtual void loadCheckPoints();
  
    double epsilon_by_frame();
};