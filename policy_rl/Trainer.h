/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Trainer.h
 * Author: wonki
 *
 * Created on January 31, 2020, 4:58 PM
 */
#pragma once

#include <ExperienceReplay.h>

class Trainer {
  public:
    int64_t batch_size = 1;
    double gamma = 0.99;
    int64_t frame_id = 0;
    std::vector<float> Action;
    ExperienceReplay buffer;
    std::vector<double> PrevState;
    std::vector<double> PostState;
    std::vector<float> actor_loss_;
    std::vector<float> critic_loss_;
    std::vector<float> rewards_;
      
    Trainer(uint64_t capacity) : buffer(capacity){};
    virtual ~Trainer(){}   
    virtual std::vector<float> act_graph(std::vector<float> &feat_matrix, std::vector<float> &adj_matrix, bool add_noise) {
      std::cout << "Trainer act_graph function" << std::endl;
      return std::vector<float>();
    }
    virtual void learn() {}
    virtual void saveCheckPoints() {};
    virtual void loadCheckPoints() {};
    
};