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
#include <cmath>
#include "OUNoise.h"
#include <Trainer.h>

class GraphConvolution : public torch::nn::Module {
  public:
    GraphConvolution(int64_t in_features, int64_t out_features) : torch::nn::Module() {
      weight = register_parameter("weight", torch::randn({in_features, out_features}));
      reset_parameters();
    }
    
    torch::Tensor forward(torch::Tensor feature, torch::Tensor adj) {
      torch::Tensor support = torch::matmul(feature, weight);
      torch::Tensor normalize_adj = adj.squeeze().sum(1).squeeze();
      std::cout << "This is = " << normalize_adj << std::endl;
     
      normalize_adj = torch::pow(normalize_adj, -0.5);
      normalize_adj = torch::diag(normalize_adj);
      normalize_adj = normalize_adj.unsqueeze(0); // [1, 37, 37]
      
      torch::Tensor adj_final = torch::matmul(torch::matmul(normalize_adj, adj), normalize_adj);

//      std::cout << "adj_final = " << adj_final << std::endl;
      torch::Tensor output = torch::matmul(adj_final, support);
      return output;
    }
    
    void reset_parameters() {
      const auto bound = 1 / sqrt(weight.size(1));
      torch::nn::init::uniform_(weight, -bound, bound); 
    }
    
    torch::Tensor weight;
};

class GraphActor : public torch::nn::Module {
  public:
    GraphActor(int64_t n_features, int64_t n_hidden, int64_t n_output, int64_t action_size, int64_t victim_size);
    torch::Tensor forward(torch::Tensor feature, torch::Tensor adj);

  private:
    int64_t victim_size_;
    std::shared_ptr<GraphConvolution> gc1;
    std::shared_ptr<GraphConvolution> gc2;
    torch::nn::Linear fc{nullptr}, output{nullptr};
};

class GraphCritic : public torch::nn::Module {
  public:
    GraphCritic(int64_t n_features, int64_t n_hidden, int64_t n_output, int64_t action_size, int64_t victim_size);
    torch::Tensor forward(torch::Tensor feature, torch::Tensor adj, torch::Tensor action);

  private:
    std::shared_ptr<GraphConvolution> gc1;
    std::shared_ptr<GraphConvolution> gc2;
    int64_t victim_size_;
    torch::nn::Linear fc{nullptr}, output{nullptr};
}; 

class DDPGTrainer : public Trainer {
  public:
    double tau = 1e-3;              // for soft update of target parameters
    double lr_actor = 1e-4;         // learning rate of the actor
    double lr_critic = 1e-4;        // learning rate of the critic
    double weight_decay = 0;        // L2 weight decay
    
    OUNoise* noise;
    
    std::shared_ptr<GraphActor> actor_local;
    std::shared_ptr<GraphActor> actor_target;
    torch::optim::Adam actor_optimizer;

    std::shared_ptr<GraphCritic> critic_local;
    std::shared_ptr<GraphCritic> critic_target;
    torch::optim::Adam critic_optimizer;
    torch::Device device;
       
  DDPGTrainer(int64_t n_features, int64_t n_hidden, int64_t n_output, int64_t action_size, int64_t victim_size, int64_t capacity);
  virtual std::vector<float> act_graph(std::vector<float> &feat_matrix, std::vector<float> &adj_matrix, bool add_noise);
  void reset() {
    noise->reset();  
  }
  virtual void learn();
  void soft_update(std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target);
  void hard_copy( std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target);
  
  virtual void saveCheckPoints();
  virtual void loadCheckPoints();
};

