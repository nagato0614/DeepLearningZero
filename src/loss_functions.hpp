//
// Created by toru on 2023/12/28.
//

#ifndef DEEPLEARNINGZERO_SRC_LOSS_FUNCTIONS_HPP_
#define DEEPLEARNINGZERO_SRC_LOSS_FUNCTIONS_HPP_

#include "nagatolib.hpp"

namespace nagato
{

template<typename Primitive>
Primitive MeanSquaredError(const MatrixN<Primitive> &y,
                           const MatrixN<Primitive> &t)
{
  auto diff = y - t;
  auto diff2 = diff * diff;
  auto sum = Sum(diff2);
  return sum / static_cast<Primitive>(2);
}

/**
 * @brief 交差エントロピー誤差
 * @tparam Primitive
 * @param y
 * @param t
 * @return
 */
template<typename Primitive>
Primitive CrossEntropyError(const MatrixN<Primitive> &y,
                            const MatrixN<Primitive> &t)
{
  constexpr auto delta = static_cast<Primitive>(1e-7);
  const auto log_y = Log(y + delta);
  const auto log_y_t = log_y * t;
  const auto sum = Sum(log_y_t);
  return -sum;
}

/**
 * @brief 交差エントロピー誤差. batch 処理対応版
 * ラベルは one-hot 表現で与える
 * @tparam Primitive
 * @param y
 * @param t
 * @return
 */
template<typename Primitive>
Primitive
CrossEntropyError(const std::vector<MatrixN<Primitive>> &y,
                  const std::vector<MatrixN<Primitive>> &t)
{
  Primitive sum = 0;

  for (std::size_t i = 0; i < y.size(); i++)
  {
    sum += -CrossEntropyError(y[i], t[i]);
  }
  return -sum / static_cast<Primitive>(y.size());
}
}
#endif //DEEPLEARNINGZERO_SRC_LOSS_FUNCTIONS_HPP_
