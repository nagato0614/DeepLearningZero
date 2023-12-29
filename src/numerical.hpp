//
// Created by toru on 2023/12/28.
//

#ifndef DEEPLEARNINGZERO_SRC_NUMERICAL_HPP_
#define DEEPLEARNINGZERO_SRC_NUMERICAL_HPP_

#include "nagatolib.hpp"

namespace nagato
{
/**
 * 数値微分
 * @tparam Primitive
 * @tparam F
 * @param f
 * @param x
 * @return
 */
template<typename Primitive, typename F>
Primitive numerical_diff(F &&f, Primitive x) noexcept
{
  constexpr Primitive h = 1e-4;
  return (f(x + h) - f(x - h)) / (2 * h);
}

/**
 * 偏微分
 * @tparam T
 * @tparam F
 * @param f
 * @param x 1次元ベクトル
 * @param idx
 * @return
 */
template<typename Primitive, typename F>
MatrixN<Primitive> gradient(F &&f, MatrixN<Primitive> &x) noexcept
{
  constexpr Primitive h = 1e-4;
  MatrixNf grad(x);

  for (std::size_t i = 0; i < x.Column(); i++)
  {
    const auto tmp_val = x[0][i];

    // f(x + h)
    x[0][i] = tmp_val + h;
    const auto fxh1 = f(x);

    // f(x - h)
    x[0][i] = tmp_val - h;
    const auto fxh2 = f(x);

    grad[0][i] = (fxh1 - fxh2) / (2 * h);
    x[0][i] = static_cast<Primitive>(tmp_val);
  }

  return grad;
}


template<typename Primitive, typename F>
MatrixN<Primitive>
gradient_descent(F &&f, MatrixN<Primitive> &init_x, Primitive lr=0.01, int step_num=100) noexcept
{
  MatrixNf x(init_x);
  for (int i = 0; i < step_num; i++)
  {
    const auto grad = numerical_diff(f, x);
    x -= lr * grad;
  }

  return x;
}

}

#endif //DEEPLEARNINGZERO_SRC_NUMERICAL_HPP_
