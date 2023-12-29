//
// Created by toru on 2023/12/28.
//

#ifndef DEEPLEARNINGZERO_SRC_RELU_HPP_
#define DEEPLEARNINGZERO_SRC_RELU_HPP_

#include "nagatolib.hpp"

namespace nagato
{
class Relu
{
 public:
  Relu();

  /**
   * @brief batch処理用
   * @param x
   * @return
   */
  std::vector<MatrixNf> Forward(const std::vector<MatrixNf> &x) noexcept;

  /**
   * @brief batch処理用
   * @param dout
   * @return
   */
  std::vector<MatrixNf> Backward(const std::vector<MatrixNf> &dout) noexcept;

  std::vector<MatrixN<bool>> mask_;
};
}

#endif //DEEPLEARNINGZERO_SRC_RELU_HPP_
