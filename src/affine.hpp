//
// Created by toru on 2023/12/28.
//

#ifndef DEEPLEARNINGZERO_SRC_AFFINE_HPP_
#define DEEPLEARNINGZERO_SRC_AFFINE_HPP_
#include "nagatolib.hpp"

namespace nagato
{
class Affine
{
 public:
  Affine(const MatrixNf &W, const MatrixNf &b) noexcept;

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

  std::vector<MatrixNf> W_;
  std::vector<MatrixNf> b_;
  MatrixNf x_;
  MatrixNf dW_;
  MatrixNf db_;
};
}
#endif //DEEPLEARNINGZERO_SRC_AFFINE_HPP_
