#include "DQNTrainer.h"
#include "DQN.h"
#include "ExperienceReplay.h"
#include <math.h>
#include <chrono>

DQNTrainer::DQNTrainer(int64_t n_feature, int64_t n_hidden, int64_t n_output, int64_t action_size, int64_t victim_size, int64_t capacity)
    : Trainer(capacity),
      dqn_local(std::make_shared<GraphDQN>(n_feature, n_hidden, n_output, victim_size)),
      dqn_target(std::make_shared<GraphDQN>(n_feature, n_hidden, n_output, victim_size)),
      dqn_optimizer(dqn_local->parameters(), lr_dqn),
      device(torch::kCPU) {
 
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        device_type = torch::kCUDA;
        std::cout << "Agent - Cuda available" << std::endl;
    } else {
        device_type = torch::kCPU;
        std::cout << "Agent - CPU used" << std::endl;
    }
    
    device = torch::Device(device_type);
    
    victim_size_ = victim_size;
    dqn_local->to(device);
    dqn_target->to(device);
    
    dqn_local->to(torch::kFloat);
    dqn_target->to(torch::kFloat);

    loadCheckPoints();     
    srand((unsigned int)time(NULL));
}  

/* Graph Actor */
GraphDQN::GraphDQN(int64_t n_feature, int64_t n_hidden, int64_t n_output, int64_t victim_size)
  : gc1(std::make_shared<GraphConvolution>(n_feature, n_hidden)),
    gc2(std::make_shared<GraphConvolution>(n_hidden, n_output)) {
  victim_size_ = victim_size;
  register_module("gc1", gc1);
  register_module("gc2", gc2);
  fc = register_module("fc", torch::nn::Linear(n_output*victim_size, n_output));
  output = register_module("output", torch::nn::Linear(n_output, victim_size));
}

torch::Tensor GraphDQN::forward(torch::Tensor feature, torch::Tensor adj) {
  torch::Tensor input = torch::relu(gc1->forward(feature, adj));

  input = torch::dropout(input, 0.3, is_training());
  input = gc2->forward(input, adj);

  input = input.slice(1, 0, victim_size_);
  input = input.view({input.size(0), -1});
  input = torch::relu(fc(input));

  input = torch::sigmoid(output(input));

  return input;
}

void DQNTrainer::learn() {
  std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> batch =
    buffer.sample_queue(batch_size);

  std::vector<torch::Tensor> prev_adj_tensor;
  std::vector<torch::Tensor> prev_feat_tensor;
  std::vector<torch::Tensor> post_adj_tensor;
  std::vector<torch::Tensor> post_feat_tensor;
  std::vector<torch::Tensor> actions;
  std::vector<torch::Tensor> rewards;

  for (auto i : batch) {
    prev_adj_tensor.push_back(std::get<0>(i));
    prev_feat_tensor.push_back(std::get<1>(i));
    post_adj_tensor.push_back(std::get<2>(i));
    post_feat_tensor.push_back(std::get<3>(i));
    actions.push_back(std::get<4>(i));
    rewards.push_back(std::get<5>(i));
  }
  
  torch::Tensor prev_adj_tensors = torch::cat(prev_adj_tensor, 0).to(device);
  torch::Tensor prev_feat_tensors = torch::cat(prev_feat_tensor, 0).to(device);
  torch::Tensor post_adj_tensors = torch::cat(post_adj_tensor, 0).to(device);
  torch::Tensor post_feat_tensors = torch::cat(post_feat_tensor, 0).to(device);
  torch::Tensor action_tensors = torch::cat(actions, 0).to(device);
  torch::Tensor reward_tensors = torch::cat(rewards, 0).to(device);

  torch::Tensor q_values = dqn_local->forward(prev_feat_tensors, prev_adj_tensors);
  torch::Tensor next_target_q_values = dqn_target->forward(post_feat_tensors, post_adj_tensors);
  torch::Tensor next_q_values = dqn_local->forward(post_feat_tensors, post_adj_tensors);

  actions_tensor = actions_tensor.to(torch::kInt64);

  torch::Tensor q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1);
  torch::Tensor maximum = std::get<1>(next_q_values.max(1));
  torch::Tensor next_q_value = next_target_q_values.gather(1, maximum.unsqueeze(1)).squeeze(1);
  torch::Tensor expected_q_value = rewards_tensor + gamma * next_q_value;
        
  torch::Tensor loss = torch::mse_loss(q_value, expected_q_value);

  dqn_optimizer.zero_grad();
  loss.backward();
  dqn_optimizer.step();

  return loss;  
}

double DQNTrainer::epsilon_by_frame() {
  return epsilon_final + (epsilon_start - epsilon_final) * exp(-1. * frame_id / epsilon_decay);
}

std::vector<float> DQNTrainer::act_graph(std::vector<float> &feat_matrix, std::vector<float> &adj_matrix, bool add_noise) {
  double epsilon = epsilon_by_frame();
  frame_id++;
  auto r = ((double) rand() / (RAND_MAX));
  if (r <= epsilon){
    std::vector<float> v();
    v.emplace_back(rand() % victim_size_);    
    return v;
  }
    
    
  torch::Tensor feat_tensor = torch::from_blob(feat_matrix.data(), {1, (long int) (feat_matrix.size()/3), 3}, torch::dtype(torch::kFloat)).to(device);
  torch::Tensor adj_tensor = torch::from_blob(adj_matrix.data(), {1, (long int) (sqrt(adj_matrix.size())), 
          (long int) (sqrt(adj_matrix.size()))}, torch::dtype(torch::kFloat)).to(device);
  torch::Tensor q_value = dqn_local->forward(feat_tensor, adj_tensor).to(torch::kCPU);
  torch::Tensor action = std::get<1>(q_value.max(1));
  std::vector<float> v(action.data_ptr<float>(), action.data_ptr<float>() + action.numel());
  return v;  
}

void DQNTrainer::hard_copy(std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target) {
  torch::NoGradGuard no_grad;
  for (size_t i = 0; i < target->parameters().size(); i++) {
    target->parameters()[i].copy_(local->parameters()[i]);
  }
}

void DQNTrainer::saveCheckPoints()
{
    auto fileActor ("/home/wonki/rsm_checkpoint/ckp_actor.pt");
    auto fileCritic ("/home/wonki/rsm_checkpoint/ckp_critic.pt");
    
    torch::save(std::dynamic_pointer_cast<torch::nn::Module>(actor_local) , fileActor);
    torch::save(std::dynamic_pointer_cast<torch::nn::Module>(critic_local) , fileCritic);
}

void DQNTrainer::loadCheckPoints()
{
    auto fileActor ("/home/wonki/rsm_checkpoint/ckp_actor.pt");
    auto fileCritic ("/home/wonki/rsm_checkpoint/ckp_critic.pt");
    struct stat actor_buffer;
    struct stat critic_buffer;
    if((stat(fileActor, &actor_buffer) == 0) && 
       (stat(fileCritic, &critic_buffer) == 0 )) {
      torch::load(actor_local, fileActor);
      torch::load(critic_local, fileCritic);
    }
}
