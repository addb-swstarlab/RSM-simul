/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   DDPGTrainer.h
 * Author: wonki
 *
 * Created on January 31, 2020, 4:25 PM
 */
#pragma once

#include <torch/torch.h>
#include <ExperienceReplay.h>

#include "OUNoise.h"
#include <Trainer.h>

class Actor : public torch::nn::Module {
  public:
    Actor(int64_t channelSize, int64_t action_size);
    torch::Tensor forward(torch::Tensor state);

  private:
//    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear linear1{nullptr}, output{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
};

class Critic : public torch::nn::Module {
  public:
    Critic(int64_t channelSize, int64_t action_size);
    torch::Tensor forward(torch::Tensor x, torch::Tensor action);

  private:
//    torch::nn::Conv1d conv1{nullptr}, conv2{nullptr};
//    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear linear1{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
}; 

class GraphConvolution : public torch::nn::Module {
  public:
    GraphConvolution(int64_t in_features, int64_t out_features) : torch::nn::Module() {
      weight = register_parameter("weight", torch::randn({in_features, out_features}));
    }
    
    torch::Tensor forward(torch::Tensor feature, torch::Tensor adj) {
      torch::Tensor support = torch::matmul(feature, weight);
      torch::Tensor output = torch::matmul(adj, support);
      return output;
    }
    torch::Tensor weight;
};

class GraphActor : public torch::nn::Module {
  public:
    GraphActor(int64_t n_features, int64_t n_hidden, int64_t n_output, int64_t action_size);
    torch::Tensor forward(torch::Tensor feature, torch::Tensor adj);

  private:
    std::shared_ptr<GraphConvolution> gc1;
    std::shared_ptr<GraphConvolution> gc2;
    torch::nn::Linear linear1{nullptr}, output{nullptr};
};

class GraphCritic : public torch::nn::Module {
  public:
    GraphCritic(int64_t n_features, int64_t n_hidden, int64_t n_output, int64_t action_size);
    torch::Tensor forward(torch::Tensor feature, torch::Tensor adj, torch::Tensor action);

  private:
    std::shared_ptr<GraphConvolution> gc1;
    std::shared_ptr<GraphConvolution> gc2;
    torch::nn::Linear linear1{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
}; 

class DDPGTrainer : public Trainer {
  public:
    double tau = 1e-3;              // for soft update of target parameters
    double lr_actor = 1e-4;         // learning rate of the actor
    double lr_critic = 1e-3;        // learning rate of the critic
    double weight_decay = 0;        // L2 weight decay
    
    OUNoise* noise;
    
    std::shared_ptr<GraphActor> actor_local;
    std::shared_ptr<GraphActor> actor_target;
    torch::optim::Adam actor_optimizer;

    std::shared_ptr<GraphCritic> critic_local;
    std::shared_ptr<GraphCritic> critic_target;
    torch::optim::Adam critic_optimizer;
    torch::Device device;
       
  DDPGTrainer(int64_t n_features, int64_t n_hidden, int64_t n_output, int64_t action_size, int64_t capacity);
  virtual std::vector<float> act_graph(std::vector<float> feat_matrix, std::vector<float> adj_matrix, bool add_noise);
  void reset() {
    noise->reset();  
  }
  virtual void learn();
  void soft_update(std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target);
  void hard_copy( std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target);
  
  virtual void saveCheckPoints();
  virtual void loadCheckPoints();
};

