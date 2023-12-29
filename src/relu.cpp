//
// Created by toru on 2023/12/28.
//

#include "relu.hpp"

namespace nagato
{

Relu::Relu()
{

}

MatrixNf Relu::Forward(const MatrixNf &x) noexcept
{
  auto func = [](float x) -> bool
  { return x <= 0; };

  this->mask_ = MatrixN<bool>(x.Row(), x.Column());
  MatrixNf out(x.Row(), x.Column());
  for (std::size_t i = 0; i < x.Row(); i++)
  {
    for (std::size_t j = 0; j < x.Column(); j++)
    {
      const bool tmp = func(x[i][j]);
      mask_[i][j] = tmp;
      out[i][j] = tmp ? 0 : x[i][j];
    }
  }
  return out;
}

std::vector<MatrixNf> Relu::Forward(const std::vector<MatrixNf> &x) noexcept
{
  std::vector<MatrixNf> out(x.size());
  auto func = [](float x) -> bool
  { return x <= 0; };
  mask_.resize(x.size());

  for (std::size_t i = 0; i < x.size(); i++)
  {
    out.at(i) = MatrixNf(x.at(i).Row(), x.at(i).Column());
    mask_.at(i) = MatrixN<bool>(x.at(i).Row(), x.at(i).Column());
    for (std::size_t j = 0; j < x.at(i).Row(); j++)
    {
      for (std::size_t k = 0; k < x.at(i).Column(); k++)
      {
        const bool tmp = func(x.at(i)[j][k]);
        mask_.at(i)[j][k] = tmp;
        out.at(i)[j][k] = tmp ? 0 : x.at(i)[j][k];
      }
    }
  }
  return out;
}

std::vector<MatrixNf> Relu::Backward(const std::vector<MatrixNf> &dout) noexcept
{
  std::vector<MatrixNf> dx(dout.size());
  for (std::size_t i = 0; i < dout.size(); i++)
  {
    const auto mask = this->mask_.at(i);
    for (std::size_t j = 0; j < dout.at(i).Row(); j++)
    {
      for (std::size_t k = 0; k < dout.at(i).Column(); k++)
      {
        dx.at(i)[j][k] = mask[j][k] ? 0 : dout.at(i)[j][k];
      }
    }
  }
  return dx;
}

}
