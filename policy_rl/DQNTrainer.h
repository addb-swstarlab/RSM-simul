#pragma once

#include <torch/torch.h>
#include "ExperienceReplay.h"
#include "DQN.h"
#include <Trainer.h>

class DQNTrainer : public Trainer {
  public:
    DQN network, target_network;
    torch::optim::Adam dqn_optimizer;
    double epsilon_start = 0.7;
    double epsilon_final = 0.01;
    int64_t epsilon_decay = 30000;
  
    DQNTrainer(int64_t input_channels, int64_t num_actions, int64_t capacity);
    torch::Tensor compute_td_loss();
    double epsilon_by_frame();
    torch::Tensor get_tensor_observation(std::vector<float> &state);
    void loadstatedict(torch::nn::Module& model, torch::nn::Module& target_model);
};
