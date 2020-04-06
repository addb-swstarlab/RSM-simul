
/* 
 * 
 * Author: wonki
 *
 * 
 */
#pragma once

class OUNoise {
private:
  size_t size;
  std::vector<float> mu;
  std::vector<float> state;
  double theta=0.15;
  //double theta=0.3;
  double sigma=0.1;

public:
  OUNoise (size_t size_in) {
    size = size_in;
    mu = std::vector<float>(size, 0);
    srand((unsigned int)time(NULL));
    reset();
  }

  void reset() {
    state = mu;
  }

  void sample(std::vector<float> &action) {
    for (size_t i = 0; i < state.size(); i++) {
      auto random = ((double) rand() / (double)RAND_MAX);
      float dx = theta * (mu[i] - state[i]) + sigma * random;
      state[i] = state[i] + dx;
      action[i] += state[i];
    }
 }
};