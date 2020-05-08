#include "DQNTrainer.h"
#include <sys/stat.h>
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
        std::cout << "Agent - DQN Cuda available" << std::endl;
    } else {
        device_type = torch::kCPU;
        std::cout << "Agent - DQN CPU used" << std::endl;
    }
    
    device = torch::Device(device_type);
//    
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
  fc = register_module("fc", torch::nn::Linear(n_output, 64));
  output = register_module("output", torch::nn::Linear(64, victim_size));
}

torch::Tensor GraphDQN::forward(torch::Tensor feature, torch::Tensor adj) {
  torch::Tensor input = torch::relu(gc1->forward(feature, adj));

  input = torch::dropout(input, 0.3, is_training());
  input = gc2->forward(input, adj);

  /* Graph embedding version */  
  input = std::get<0>(input.max(1));
  input = input.view({input.size(0), -1});
  input = torch::relu(fc(input));

  /* Node embedding version */  
//  input = input.slice(1, 0, victim_size_);
//  input = input.view({input.size(0), -1});
//  input = torch::relu(fc(input));

  input = output(input);

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
  
  /* State */
  torch::Tensor prev_adj_tensors = torch::cat(prev_adj_tensor, 0).to(device);
  torch::Tensor prev_feat_tensors = torch::cat(prev_feat_tensor, 0).to(device);
  /* State Prime */
  torch::Tensor post_adj_tensors = torch::cat(post_adj_tensor, 0).to(device);
  torch::Tensor post_feat_tensors = torch::cat(post_feat_tensor, 0).to(device);
  /* Action */
  torch::Tensor action_tensors = torch::cat(actions, 0).to(device);
  /* Reward */
  torch::Tensor reward_tensors = torch::cat(rewards, 0).to(device);

  action_tensors = action_tensors.to(torch::kInt64);

  torch::Tensor current_q_value = dqn_local->forward(prev_feat_tensors, prev_adj_tensors).gather(1, action_tensors);
  torch::Tensor max_q_prime = (std::get<0>(dqn_target->forward(post_feat_tensors, post_adj_tensors).max(1))).unsqueeze(1);
  torch::Tensor expected_q_value = reward_tensors + gamma * max_q_prime;
          
  torch::Tensor loss = torch::mse_loss(current_q_value, expected_q_value.detach());
  //loss_.emplace_back(loss.to(torch::kCPU).item<float>());
  
  dqn_optimizer.zero_grad();
  loss.backward();
  dqn_optimizer.step();
  
  if(frame_id % 500 == 0) hard_copy(dqn_local, dqn_target);
}

double DQNTrainer::epsilon_by_frame() {
  return epsilon_final + (epsilon_start - epsilon_final) * exp(-1. * frame_id / epsilon_decay);
}

int64_t DQNTrainer::act_dqn(std::vector<float> &feat_matrix, std::vector<float> &adj_matrix) {
  double epsilon = epsilon_by_frame();
  frame_id++;
  auto r = ((double) rand() / (RAND_MAX));
  
//  std::cout << std::setprecision(16) << "FRAME : " << frame_id - 1
//          << " EPSILON : " << epsilon << " r : " << r << std::endl;
  if (r <= epsilon){
    return ((int64_t)(rand() % victim_size_));    
  }
    
    
  torch::Tensor feat_tensor = torch::from_blob(feat_matrix.data(), {1, (long int) (feat_matrix.size()/3), 3}, torch::dtype(torch::kFloat)).to(device);
  torch::Tensor adj_tensor = torch::from_blob(adj_matrix.data(), {1, (long int) (sqrt(adj_matrix.size())), 
          (long int) (sqrt(adj_matrix.size()))}, torch::dtype(torch::kFloat)).to(device);
                   
  torch::Tensor q_value = dqn_local->forward(feat_tensor, adj_tensor);
  torch::Tensor action = std::get<1>(q_value.max(1));

  return action[0].to(torch::kCPU).item<int64_t>();  
}

void DQNTrainer::hard_copy(std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target) {
  torch::NoGradGuard no_grad;
  for (size_t i = 0; i < target->parameters().size(); i++) {
    target->parameters()[i].copy_(local->parameters()[i]);
  }
}

void DQNTrainer::saveCheckPoints()
{
    auto fileDQN ("/home/wonki/rsm_checkpoint/ckp_dqn.pt");
    
    torch::save(std::dynamic_pointer_cast<torch::nn::Module>(dqn_local), fileDQN);
}

void DQNTrainer::loadCheckPoints()
{
    auto fileDQN ("/home/wonki/rsm_checkpoint/ckp_dqn.pt");

    struct stat dqn_buffer;
    if((stat(fileDQN, &dqn_buffer) == 0)) {
      torch::load(dqn_local, fileDQN);
    }
}
