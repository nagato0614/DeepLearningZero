//
// Created by toru on 2023/12/28.
//

#ifndef DEEPLEARNINGZERO_SRC_SIGMOID_HPP_
#define DEEPLEARNINGZERO_SRC_SIGMOID_HPP_

#include "nagatolib.hpp"

namespace nagato
{
class Sigmoid
{
 public:
  Sigmoid();

  MatrixNf Forward(const MatrixNf &x) noexcept;

  /**
   * @brief batch 処理対応
   * @param dout
   * @return
   */
  std::vector<MatrixNf> Forward(const std::vector<MatrixNf> &x) noexcept;

  MatrixNf Backward(const MatrixNf &dout) noexcept;

  /**
   * @brief batch 処理対応
   * @param dout
   * @return
   */
  std::vector<MatrixNf> Backward(const std::vector<MatrixNf> &dout) noexcept;

  std::vector<MatrixNf> out_;
};
}

#endif //DEEPLEARNINGZERO_SRC_SIGMOID_HPP_
