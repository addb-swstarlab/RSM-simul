/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <math.h>
#include <sys/stat.h>
#include "DDPGTrainer.h"

/* Actor */
Actor::Actor(int64_t channelSize, int64_t action_size) : torch::nn::Module() {
  conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(channelSize, 32, {2, 32}).stride({1, 8})));
  conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, {2, 2}).stride({1, 1})));
 // conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, {2, 256}).stride(1)));
  //linear1 = register_module("linear1", torch::nn::Linear(64*2*254, 64));
  linear1 = register_module("linear1", torch::nn::Linear(64*2*508, 64));
  output = register_module("output", torch::nn::Linear(64, action_size));
  bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));
}

torch::Tensor Actor::forward(torch::Tensor input) {
  //input = torch::relu(bn1(conv1(input)));
  input = torch::relu(conv1(input));
  input = torch::relu(conv2(input));
//  input = torch::relu(conv3(input));

  input = input.view({input.size(0), -1});
  input = torch::relu(linear1(input));
  input = output(input);
  input = torch::sigmoid(input);
  //input = torch::tanh(input);

  return input;
}

/* Critic */
Critic::Critic(int64_t channelSize, int64_t action_size) : torch::nn::Module() {
  conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(channelSize, 32, {2, 32}).stride({1, 8})));
  conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, {2, 2}).stride({1, 1})));
 // conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, {2, 256}).stride(1)));
  //linear1 = register_module("linear1", torch::nn::Linear(64*2*254, 64));
  linear1 = register_module("linear1", torch::nn::Linear(64*2*508, 64));
  
  fc1 = register_module("fc1", torch::nn::Linear(64 + action_size, 32));
  fc2 = register_module("fc2", torch::nn::Linear(32, action_size));
  bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));
}

torch::Tensor Critic::forward(torch::Tensor input, torch::Tensor action) {
  //input = torch::relu(bn1(conv1(input)));
  input = torch::relu(conv1(input));
  input = torch::relu(conv2(input));
 // input = torch::relu(conv3(input));

  input = input.view({input.size(0), -1});
  input = torch::relu(linear1(input));
  
  auto x = torch::cat({input, action}, 1);
  x = torch::relu(fc1->forward(x));

  return fc2->forward(x);
}

/* Graph Actor */
GraphActor::GraphActor(int64_t n_feature, int64_t n_hidden, int64_t n_output, int64_t action_size)
  : gc1(std::make_shared<GraphConvolution>(n_feature, n_hidden)),
    gc2(std::make_shared<GraphConvolution>(n_hidden, n_output)){
  register_module("gc1", gc1);
  register_module("gc2", gc2);
  linear1 = register_module("linear1", torch::nn::Linear(64*2*508, 64));
  output = register_module("output", torch::nn::Linear(64, action_size));
}

torch::Tensor GraphActor::forward(torch::Tensor feature, torch::Tensor adj) {
  torch::Tensor input = torch::relu(gc1->forward(feature, adj));
  input = torch::dropout(input, 0.3, is_training());
  input = gc2->forward(feature, adj);
 
  input = input.view({input.size(0), -1});
  input = torch::relu(linear1(input));
  input = output(input);
  input = torch::sigmoid(input);
  //input = torch::tanh(input);

  return input;
}

/* Critic */
GraphCritic::GraphCritic(int64_t n_feature, int64_t n_hidden, int64_t n_output, int64_t action_size) 
  : gc1(std::make_shared<GraphConvolution>(n_feature, n_hidden)),
    gc2(std::make_shared<GraphConvolution>(n_hidden, n_output)){
  register_module("gc1", gc1);
  register_module("gc2", gc2);
  linear1 = register_module("linear1", torch::nn::Linear(64*2*508, 64));  
  fc1 = register_module("fc1", torch::nn::Linear(64 + action_size, 32));
  fc2 = register_module("fc2", torch::nn::Linear(32, action_size));
}

torch::Tensor GraphCritic::forward(torch::Tensor feature, torch::Tensor adj, torch::Tensor action) {
  torch::Tensor input = torch::relu(gc1->forward(feature, adj));
  input = torch::dropout(input, 0.3, is_training());
  input = gc2->forward(input, adj);

  input = input.view({input.size(0), -1});
  input = torch::relu(linear1(input));
  
  auto x = torch::cat({input, action}, 1);
  x = torch::relu(fc1->forward(x));

  return fc2->forward(x);
}

DDPGTrainer::DDPGTrainer(int64_t n_feature, int64_t n_hidden, int64_t n_output, int64_t action_size, int64_t capacity)
    : Trainer(capacity),
      actor_local(std::make_shared<GraphActor>(n_feature, n_hidden, n_output, action_size)),
      actor_target(std::make_shared<GraphActor>(n_feature, n_hidden, n_output, action_size)),
      actor_optimizer(actor_local->parameters(), lr_actor),
      critic_local(std::make_shared<GraphCritic>(n_feature, n_hidden, n_output, action_size)),
      critic_target(std::make_shared<GraphCritic>(n_feature, n_hidden, n_output, action_size)),
      critic_optimizer(critic_local->parameters(), lr_critic),
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
    
    actor_local->to(device);
    actor_target->to(device);
    
    actor_local->to(torch::kFloat);
    actor_target->to(torch::kFloat);

    critic_local->to(device);
    critic_target->to(device);
    
    critic_local->to(torch::kFloat);
    critic_target->to(torch::kFloat);

    //critic_optimizer.options.weight_decay_ = weight_decay;
    
    hard_copy(actor_target, actor_local);
    hard_copy(critic_target, critic_local);
    noise = new OUNoise(static_cast<size_t>(action_size));
    
    loadCheckPoints();   
    PrevState.reserve(49152);
    PostState.reserve(49152);
}  

std::vector<float> DDPGTrainer::act_graph(std::vector<uint32_t> adj_matrix, std::vector<float> feat_matrix, bool add_noise) {
  torch::Tensor adj_tensor = torch::from_blob(adj_matrix.data(), {1, 10000, 10000}, torch::dtype(torch::kInt32)).to(device);
  torch::Tensor feat_tensor = torch::from_blob(feat_matrix.data(),{1, 10000, 3}, torch::dtype(torch::kFloat)).to(device);
  //torch::Tensor torchState = torch::from_blob(state.data(), {1,4,4,256}, torch::dtype(torch::kDouble));
  actor_local->eval();

  torch::NoGradGuard guard;
  torch::Tensor action = actor_local->forward(adj_tensor, feat_tensor).to(torch::kCPU);

  actor_local->train();

  std::vector<float> v(action.data_ptr<float>(), action.data_ptr<float>() + action.numel());
  //for(int i = 0; i < 1; i++)  std::cout << "prev action = " << v[i] << std::endl;
  if(add_noise) noise->sample(v);
  
//  for(int i = 0; i < 8; i++)  std::cout << "after action = " << v[i] << std::endl;
  
  for (size_t i = 0; i < v.size(); i++) {
    v[i] = std::fmin(std::fmax(v[i], 0.f), 1.f); // 0 =< v[i] =< 1
  }
  
//  for(int i = 0; i < 8; i++)   std::cout << "last action = " << v[i] << std::endl;
  return v;
}

void DDPGTrainer::learn() {

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
                   
  auto actions_next = actor_target->forward(post_adj_tensors, post_feat_tensors);
  auto Q_targets_next = critic_target->forward(post_adj_tensors, post_feat_tensors, actions_next);
  auto Q_targets = reward_tensors + (gamma * Q_targets_next);
  auto Q_expected = critic_local->forward(prev_adj_tensors, prev_feat_tensors, action_tensors); 

  torch::Tensor critic_loss = torch::mse_loss(Q_expected, Q_targets.detach());
  critic_loss_.push_back(critic_loss.to(torch::kCPU).item<float>());
  //std::cout << "CRITIC_LOSS = " << critic_loss.to(torch::kCPU).item<double>() << std::endl;
  critic_optimizer.zero_grad();
  critic_loss.backward();
  critic_optimizer.step();

  auto actions_pred = actor_local->forward(prev_adj_tensors, prev_feat_tensors);
  auto actor_loss = -critic_local->forward(prev_adj_tensors, prev_feat_tensors, actions_pred).mean();
  actor_loss_.push_back(actor_loss.to(torch::kCPU).item<float>());
  //std::cout << "ACTOR_LOSS = " << actor_loss.to(torch::kCPU).item<double>() << std::endl;

  actor_optimizer.zero_grad();
  actor_loss.backward();
  actor_optimizer.step();

  soft_update(critic_local, critic_target);
  soft_update(actor_local, actor_target); 
}

void DDPGTrainer::soft_update(std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target) {
  torch::NoGradGuard no_grad;
  for (size_t i = 0; i < target->parameters().size(); i++) {
    target->parameters()[i].copy_(tau * local->parameters()[i] + (1.0 - tau) * target->parameters()[i]);
  }   
}

void DDPGTrainer::hard_copy(std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target) {
  for (size_t i = 0; i < target->parameters().size(); i++) {
    target->parameters()[i] = local->parameters()[i];
  }
}

void DDPGTrainer::saveCheckPoints()
{
    auto fileActor ("/home/wonki/rsm_checkpoint/ckp_actor.pt");
    auto fileCritic ("/home/wonki/rsm_checkpoint/ckp_critic.pt");
    
    torch::save(std::dynamic_pointer_cast<torch::nn::Module>(actor_local) , fileActor);
    torch::save(std::dynamic_pointer_cast<torch::nn::Module>(critic_local) , fileCritic);
}

void DDPGTrainer::loadCheckPoints()
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
