/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include "NDQNTrainer.h"
#include <sys/stat.h>
#include "ExperienceReplay.h"
#include <math.h>
#include <chrono>

NDQNTrainer::NDQNTrainer(int64_t action_size, int64_t capacity)
  : Trainer(capacity),
    dqn_local(std::make_shared<DQN>(action_size)),
    dqn_target(std::make_shared<DQN>(action_size)),
    dqn_optimizer(dqn_local->parameters(), torch::optim::AdamOptions(0.0005).beta1(0.5)),
    device(torch::kCPU) {
 
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
      device_type = torch::kCUDA;
      std::cout << "Agent - DQN Cuda available" << std::endl;
  } else {
      device_type = torch::kCPU;
      std::cout << "Agent - DQN CPU used" << std::endl;
  }
    
  device = torch::Device(device_type);

  dqn_local->to(device);
  dqn_target->to(device);
    
  dqn_local->to(torch::kFloat);
  dqn_target->to(torch::kFloat);
  action_size_ = action_size;
  srand((unsigned int)time(NULL));
   
}  

void NDQNTrainer::learn() {
  std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> batch =
    buffer.sample_queue(batch_size);

  std::vector<torch::Tensor> prev_states;
  std::vector<torch::Tensor> new_states;
  std::vector<torch::Tensor> actions;
  std::vector<torch::Tensor> rewards;
  
  for (auto i : batch) {
    prev_states.push_back(std::get<0>(i));
    new_states.push_back(std::get<1>(i));
    actions.push_back(std::get<2>(i));
    rewards.push_back(std::get<3>(i));
  }
  
  torch::Tensor prev_states_tensor;
  torch::Tensor new_states_tensor;
  torch::Tensor actions_tensor;
  torch::Tensor rewards_tensor;

  prev_states_tensor = torch::cat(prev_states, 0).to(device);
  new_states_tensor = torch::cat(new_states, 0).to(device);
  actions_tensor = torch::cat(actions, 0).to(device);
  rewards_tensor = torch::cat(rewards, 0).to(device);

  actions_tensor = actions_tensor.to(torch::kInt64);
  
  torch::Tensor current_q_value = dqn_local->forward(prev_states_tensor).gather(1, actions_tensor);
  torch::Tensor max_q_prime = (std::get<0>(dqn_target->forward(new_states_tensor).max(1))).unsqueeze(1);
  torch::Tensor expected_q_value = rewards_tensor + gamma * max_q_prime;

  torch::Tensor loss = torch::mse_loss(current_q_value, expected_q_value.detach());
  loss_.emplace_back(loss.to(torch::kCPU).item<float>());

  dqn_optimizer.zero_grad();
  loss.backward();
  dqn_optimizer.step();
  if(frame_id % 1000 == 0) hard_copy(dqn_local, dqn_target);
}

double NDQNTrainer::epsilon_by_frame() {
  return epsilon_final + (epsilon_start - epsilon_final) * exp(-1. * frame_id / epsilon_decay);
}

int64_t NDQNTrainer::act(torch::Tensor state) {
  frame_id++;
  double epsilon = epsilon_by_frame();
  auto r = ((double) rand() / (RAND_MAX));
  
  if(frame_id % 1000 == 0)
    std::cout << "epsilon : " << epsilon << " r : " << r << std::endl; 

  if (r <= epsilon) {
    return ((int64_t)(rand() % action_size_));    
  }
  
  state = state.to(device);                         
  torch::Tensor action_tensor = dqn_local->act(state);
  return action_tensor[0].to(torch::kCPU).item<int64_t>();  
}

void NDQNTrainer::hard_copy(std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target) {
  torch::NoGradGuard no_grad;
  for (size_t i = 0; i < target->parameters().size(); i++) {
    target->parameters()[i].copy_(local->parameters()[i]);
  }
}

void NDQNTrainer::saveCheckPoints() {
  auto fileDQN ("/home/wonki/rsm_checkpoint/ckp_dqn.pt");    
  torch::save(std::dynamic_pointer_cast<torch::nn::Module>(dqn_local), fileDQN);
}

void NDQNTrainer::loadCheckPoints() {
  auto fileDQN ("/home/wonki/rsm_checkpoint/ckp_dqn.pt");
  struct stat dqn_buffer;
  if((stat(fileDQN, &dqn_buffer) == 0)) {
    torch::load(dqn_local, fileDQN);
  }
}

