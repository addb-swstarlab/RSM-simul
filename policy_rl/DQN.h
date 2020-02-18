#pragma once

#include <torch/torch.h>

struct DQN : torch::nn::Module{
  DQN(int64_t input_channels, int64_t num_actions)
    :
    conv1(torch::nn::Conv1dOptions(input_channels, 32, 2).stride(1)),
    conv2(torch::nn::Conv1dOptions(32, 64, 2).stride(1)),
//            conv3(torch::nn::Conv2dOptions(64, 64, 3)
//                          .stride(1)
//                          ),
    linear1(torch::nn::Linear(64*2, 64)),
    output(torch::nn::Linear(64, num_actions)){}

torch::Tensor forward(torch::Tensor input) {
  input = input.transpose(1, 2);
  input = torch::relu(conv1(input));
  input = torch::relu(conv2(input));
//        input = torch::relu(conv3(input));

  // Flatten the output
  input = input.view({input.size(0), -1});
  input = torch::relu(linear1(input));
  input = output(input);

  return input;
}

torch::Tensor act(torch::Tensor state){
  torch::Tensor q_value = forward(state);
  torch::Tensor action = std::get<1>(q_value.max(1));
  return action;
}
 //conv3
  torch::nn::Conv1d conv1, conv2;
  torch::nn::Linear linear1, output;
};
