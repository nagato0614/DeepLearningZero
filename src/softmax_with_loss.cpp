//
// Created by toru on 2023/12/28.
//

#include "softmax_with_loss.hpp"
#include "loss_functions.hpp"
#include "activation_func.hpp"

namespace nagato
{

SoftmaxWithLoss::SoftmaxWithLoss()
{

}

float SoftmaxWithLoss::Forward(const MatrixNf &x, const MatrixNf &t) noexcept
{
  y_ = Softmax(x);
  loss_ = CrossEntropyError(y_, t);
  return loss_;
}

MatrixNf SoftmaxWithLoss::Backward(const MatrixNf &dout) noexcept
{
  const auto batch_size = this->t_.Row();
  const auto dx = (y_ - t_) / batch_size;
  return dx;
}

}