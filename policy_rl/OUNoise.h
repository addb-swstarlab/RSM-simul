
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
  std::vector<double> mu;
  std::vector<double> state;
  double theta=0.15;
  double sigma=0.1;

public:
  OUNoise (size_t size_in) {
    size = size_in;
    mu = std::vector<double>(size, 0);
    reset();
  }

  void reset() {
    state = mu;
  }

  void sample(std::vector<double> &action) {
    srand((unsigned int)time(NULL));
    for (size_t i = 0; i < state.size(); i++) {
      auto random = ((double) rand() / (RAND_MAX));
      float dx = theta * (mu[i] - state[i]) + sigma * random;
      state[i] = state[i] + dx;
      action[i] += state[i];
    }
 }
};