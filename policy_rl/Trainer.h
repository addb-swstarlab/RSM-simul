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
    int64_t batch_size = 128;
    double gamma = 0.99;
    int64_t frame_id = 0;
    std::vector<double> Action;
    ExperienceReplay buffer;
    std::vector<double> PrevState;
    std::vector<double> PostState;
    std::vector<double> actor_loss_;
    std::vector<double> critic_loss_;
    std::vector<double> rewards_;
      
    Trainer(uint64_t capacity) : buffer(capacity){};
    virtual ~Trainer(){}
    virtual std::vector<double> act(std::vector<double> state, bool add_noise) {
      std::cout << "Trainer act function" << std::endl;
      return std::vector<double>();
    }
    virtual void learn() {}
    virtual void saveCheckPoints() {};
    virtual void loadCheckPoints() {};
    
};