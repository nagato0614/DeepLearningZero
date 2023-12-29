//
// Created by toru on 2023/12/28.
//

#include "simple_net.hpp"
#include "random.hpp"
#include "activation_func.hpp"
#include "loss_functions.hpp"

namespace nagato
{

SimpleNet::SimpleNet()
{
  W_ = Randn<float>(2, 3);
  b_ = Randn<float>(1, 3);
}

MatrixNf SimpleNet::predict(const MatrixNf &x) const noexcept
{
  return Dot(x, W_);
}

float SimpleNet::loss(const MatrixNf &x, const MatrixNf &t) const noexcept
{
  auto z = predict(x);
  auto y = Softmax(z);
  auto loss = CrossEntropyError(y, t);
  return loss;
}

} // namespace nagato