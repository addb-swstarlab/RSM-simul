#pragma once

#include <torch/torch.h>
#include "ExperienceReplay.h"
#include "DQN.h"
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
     
      normalize_adj = torch::pow(normalize_adj, -0.5);
      normalize_adj = torch::diag(normalize_adj);
      normalize_adj = normalize_adj.unsqueeze(0); 
      
      torch::Tensor adj_final = torch::matmul(torch::matmul(normalize_adj, adj), normalize_adj);

      torch::Tensor output = torch::matmul(adj_final, support);
      return output;
    }
    
    void reset_parameters() {
      const auto bound = 1 / sqrt(weight.size(1));
      torch::nn::init::uniform_(weight, -bound, bound); 
    }
    
    torch::Tensor weight;
};

class GraphDQN : public torch::nn::Module {
  public:
    GraphDQN(int64_t n_features, int64_t n_hidden, int64_t n_output, int64_t victim_size);
    torch::Tensor forward(torch::Tensor feature, torch::Tensor adj);

  private:
    int64_t victim_size_;
    std::shared_ptr<GraphConvolution> gc1;
    std::shared_ptr<GraphConvolution> gc2;
    torch::nn::Linear fc{nullptr}, output{nullptr};
};


class DQNTrainer : public Trainer {
  public:
    double epsilon_start = 0.7;
    double epsilon_final = 0.01;
    int64_t epsilon_decay = 30000;
    int64_t victim_size_;
    
    double lr_dqn = 1e-4;         // learning rate of the actor
    double weight_decay = 0;        // L2 weight decay
        
    std::shared_ptr<GraphActor> dqn_local;
    std::shared_ptr<GraphActor> dqn_target;
    torch::optim::Adam dqn_optimizer;
    torch::Device device;
       
    DQNTrainer(int64_t n_features, int64_t n_hidden, int64_t n_output, int64_t action_size, int64_t victim_size, int64_t capacity);
    virtual std::vector<float> act_graph(std::vector<float> &feat_matrix, std::vector<float> &adj_matrix, bool add_noise);
    virtual void learn();
    void hard_copy( std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target);
  
    virtual void saveCheckPoints();
    virtual void loadCheckPoints();
  
    double epsilon_by_frame();

};
