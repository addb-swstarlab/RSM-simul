#include "util.h"
#include <cstdlib>
#include <cstdio>

template <typename T>
void sequence(T n, std::vector<T>& out) {
  out.clear();
  out.reserve(static_cast<std::size_t>(n));
  for (T i = 0; i < n; i++) out.push_back(i);
}

template <typename T>
void shuffle(std::vector<T>& v) {
  unsigned int seed =
      0;  // std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle(v.begin(), v.end(), std::default_random_engine(seed));
  // std::size_t count = v.size();
  // for (std::size_t i = 0; i < count; i++) {
  //	std::size_t j = i + (rand() % (count - i));
  //	std::swap(v[i], v[j]);
  // }
}

template void sequence(uint64_t n, std::vector<uint64_t>& out);
template void shuffle(std::vector<uint64_t>& v);
template void sequence(uint32_t n, std::vector<uint32_t>& out);
template void shuffle(std::vector<uint32_t>& v);

void uniform_pdf(uint64_t n, std::vector<double>& out_pdf) {
  out_pdf.clear();
  out_pdf.reserve(n);
  for (uint64_t i = 0; i < n; i++) out_pdf.push_back(1.);
}

void pdf_to_cdf(const std::vector<double>& pdf, std::vector<double>& out_cdf) {
  std::size_t count = pdf.size();
  out_cdf.clear();
  out_cdf.reserve(count);
  double s = 0.;
  for (std::size_t i = 0; i < count; i++) {
    s += pdf[i];
    out_cdf.push_back(s);
  }
}

// def sample(cdf, count):
//     """Gets samples from CDF."""
//     r = random.random
//     b = bisect.bisect_left
//     s = cdf[-1]
//     result = [0] * count
//     for i in range(count):
//         v = r() * s
//         k = b(cdf, v)
//         result[i] = k
//     return result
