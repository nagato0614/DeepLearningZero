//
// Created by toru on 2023/12/28.
//

#ifndef DEEPLEARNINGZERO_SRC_SOFTMAX_WITH_LOSS_HPP_
#define DEEPLEARNINGZERO_SRC_SOFTMAX_WITH_LOSS_HPP_

#include "nagatolib.hpp"

namespace nagato
{
class SoftmaxWithLoss
{
 public:
  SoftmaxWithLoss();

  float Forward(const MatrixNf &x, const MatrixNf &t) noexcept;

  MatrixNf Backward(const MatrixNf &dout) noexcept;

  MatrixNf y_;
  MatrixNf t_;
  float loss_;

};
}

#endif //DEEPLEARNINGZERO_SRC_SOFTMAX_WITH_LOSS_HPP_
