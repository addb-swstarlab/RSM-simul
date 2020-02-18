#pragma once

#include "common.h"
#include <random>
#include <array>

// A sequence of increasing numbers.
template <typename T>
void sequence(T n, std::vector<T>& out);

// In-place shuffling.
template <typename T>
void shuffle(std::vector<T>& v);

// PDF of uniform distribution
void uniform_pdf(uint64_t n, std::vector<double>& out_pdf);

// Convert PDF to CDF.
void pdf_to_cdf(const std::vector<double>& pdf, std::vector<double>& out_cdf);

// Fast random number generators.
static uint32_t fast_rand(uint64_t* state) {
  // same as Java's
  *state = (*state * 0x5deece66dUL + 0xbUL) & ((1UL << 48) - 1);
  return (uint32_t)(*state >> (48 - 32));
}

static double fast_rand_d(uint64_t* state) {
  *state = (*state * 0x5deece66dUL + 0xbUL) & ((1UL << 48) - 1);
  return (double)*state / (double)((1UL << 48) - 1);
}
