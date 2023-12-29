//
// Created by toru on 2023/12/28.
//

#include "sigmoid.hpp"

namespace nagato
{

Sigmoid::Sigmoid()
{

}

std::vector<MatrixNf> Sigmoid::Forward(const std::vector<MatrixNf> &x) noexcept
{
  std::vector<MatrixNf> out;
  auto func = [](float x) -> float
  { return 1 / (1 + std::exp(-x)); };
  out.reserve(x.size());
  for (auto i = 0uz; i < x.size(); ++i)
  {
    out.at(i) = x.at(i).Itor(func);
  }
  out_ = out;
  return out_;
}

std::vector<MatrixNf> Sigmoid::Backward(const std::vector<MatrixNf> &dout) noexcept
{
  std::vector<MatrixNf> dx;
  dx.reserve(dout.size());
  for (auto i = 0uz; i < dout.size(); ++i)
  {
    dx.at(i) = dout[i] * (1.0f - out_[i]) * out_[i];
  }
  return dx;
}

}