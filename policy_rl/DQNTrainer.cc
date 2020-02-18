#include "DQNTrainer.h"
#include "DQN.h"
#include "ExperienceReplay.h"
#include <math.h>
#include <chrono>

DQNTrainer::DQNTrainer(int64_t input_channels, int64_t num_actions, int64_t capacity): 
  Trainer(capacity),
  network(input_channels, num_actions),
  target_network(input_channels, num_actions),
  dqn_optimizer(network.parameters(), torch::optim::AdamOptions(0.0001).beta1(0.5)){}

torch::Tensor DQNTrainer::compute_td_loss() {
  std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> batch =
  buffer.sample_queue(batch_size);

  std::vector<torch::Tensor> states;
  std::vector<torch::Tensor> new_states;
  std::vector<torch::Tensor> actions;
  std::vector<torch::Tensor> rewards;

  for (auto i : batch) {
    states.push_back(std::get<0>(i));
    new_states.push_back(std::get<1>(i));
    actions.push_back(std::get<2>(i));
    rewards.push_back(std::get<3>(i));
  }

  torch::Tensor states_tensor;
  torch::Tensor new_states_tensor;
  torch::Tensor actions_tensor;
  torch::Tensor rewards_tensor;
             
  states_tensor = torch::cat(states, 0);
  new_states_tensor = torch::cat(new_states, 0);
  actions_tensor = torch::cat(actions, 0);
  rewards_tensor = torch::cat(rewards, 0);

  torch::Tensor q_values = network.forward(states_tensor);
  torch::Tensor next_target_q_values = target_network.forward(new_states_tensor);
  torch::Tensor next_q_values = network.forward(new_states_tensor);

  actions_tensor = actions_tensor.to(torch::kInt64);

  torch::Tensor q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1);
  torch::Tensor maximum = std::get<1>(next_q_values.max(1));
  torch::Tensor next_q_value = next_target_q_values.gather(1, maximum.unsqueeze(1)).squeeze(1);
  torch::Tensor expected_q_value = rewards_tensor + gamma*next_q_value;
        
  torch::Tensor loss = torch::mse_loss(q_value, expected_q_value);

  dqn_optimizer.zero_grad();
  loss.backward();
  dqn_optimizer.step();

  return loss;

}

double DQNTrainer::epsilon_by_frame() {
  return epsilon_final + (epsilon_start - epsilon_final) * exp(-1. * frame_id / epsilon_decay);
}

torch::Tensor DQNTrainer::get_tensor_observation(std::vector<float> &state) {
  torch::Tensor state_tensor = torch::from_blob(state.data(), {1, 4, 4096});
  return state_tensor;
}

void DQNTrainer::loadstatedict(torch::nn::Module& model, torch::nn::Module& target_model) {
  torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
  auto new_params = target_model.named_parameters(); // implement this
  auto params = model.named_parameters(true /*recurse*/);
  auto buffers = model.named_buffers(true /*recurse*/);
  for (auto& val : new_params) {
      auto name = val.key();
      auto* t = params.find(name);
      if (t != nullptr) {
        t->copy_(val.value());
      } else {
        t = buffers.find(name);
        if (t != nullptr) {
          t->copy_(val.value());
        }
      }
  }
}
