//
// Created by toru on 2023/12/28.
//

#include "affine.hpp"

namespace nagato
{

Affine::Affine(const MatrixNf &W, const MatrixNf &b) noexcept
  : W_(W), b_(b)
{

}

MatrixNf Affine::Forward(const MatrixNf &x) noexcept
{
  this->x_ = x;
  const auto out = Dot(x, W_) + b_;
  return out;
}

std::vector<MatrixNf> Affine::Forward(const std::vector<MatrixNf> &x) noexcept
{
  std::vector<MatrixNf> out;
  for (const auto &x_i : x)
  {
    out.emplace_back(Forward(x_i));
  }
  return out;
}

MatrixNf Affine::Backward(const MatrixNf &dout) noexcept
{
  const auto W_t = W_.Transposed();
  const auto x_t = x_.Transposed();
  const auto dx = Dot(dout, W_t);
  dW_ = Dot(x_t, dout);
  db_ = Sum(dout);
  return dx;
}

std::vector<MatrixNf> Affine::Backward(const std::vector<MatrixNf> &dout) noexcept
{
  std::vector<MatrixNf> dx;
  for (const auto &dout_i : dout)
  {
    dx.emplace_back(Backward(dout_i));
  }
  return dx;
}

}