//
// Created by toru on 2023/12/28.
//

#ifndef DEEPLEARNINGZERO_NAGATOLIB_SRC_SIMPLE_NET_HPP_
#define DEEPLEARNINGZERO_NAGATOLIB_SRC_SIMPLE_NET_HPP_

#include "nagatolib.hpp"

namespace nagato
{

class SimpleNet
{
 public:
  SimpleNet();

  MatrixNf predict(const MatrixNf &x) const noexcept;

  float loss(const MatrixNf &x, const MatrixNf &t) const noexcept;

  MatrixNf gradient(const MatrixNf &x, const MatrixNf &t) const noexcept;

  MatrixNf W_;
  MatrixNf b_;
};

}

#endif //DEEPLEARNINGZERO_NAGATOLIB_SRC_SIMPLE_NET_HPP_
