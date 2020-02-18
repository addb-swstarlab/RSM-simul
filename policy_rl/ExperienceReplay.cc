
#include "ExperienceReplay.h"
#include <memory>
#include <vector>
#include <iostream>

#include <torch/torch.h>
#include <c10/util/ArrayRef.h>
#include <algorithm>
#include <iterator>
#include <random>

ExperienceReplay::ExperienceReplay(int64_t size) {
  capacity = size;
}

void ExperienceReplay::push(torch::Tensor state,torch::Tensor new_state,
        torch::Tensor action, torch::Tensor reward) {
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample (state, new_state, action, reward);
  if (buffer.size() < capacity) {
    buffer.push_back(sample);
  } else {
    while (buffer.size() >= capacity) {
      buffer.pop_front();
    }
    buffer.push_back(sample);
  }
}

std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> ExperienceReplay::sample_queue(
            int64_t batch_size) {
  // std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> b(batch_size);
  std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> b;

  std::random_device rd;
  std::mt19937 random(rd());
  std::uniform_int_distribution<uint> range(0, buffer.size()-1);
  std::vector<uint> indices;
  uint num_entries = 0;
        
  for(;;) {
    if(num_entries == batch_size) break;
    uint idx = range(random);
    if ( (find(indices.begin(), indices.end(), idx)) == indices.end() ) {
      indices.push_back(idx);
      b.push_back(buffer[idx]);
      num_entries++;
    }    
  }
  return b;
}

int64_t ExperienceReplay::size_buffer() {
  return buffer.size();
}
