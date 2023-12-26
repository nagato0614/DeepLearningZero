//
// Created by toru on 2023/12/26.
//

#ifndef DEEPLEARNINGZERO_SRC_ACTIVATION_FUNC_HPP_
#define DEEPLEARNINGZERO_SRC_ACTIVATION_FUNC_HPP_

#include <type_traits>
#include "nagatolib.hpp"

namespace nagato
{

template<typename Primitive, std::size_t size>
constexpr Vector<Primitive, size> step_function(const Vector<Primitive, size> &x)
{
  static_assert(std::is_arithmetic_v<Primitive>);
  auto func = [](Primitive x) -> Primitive
  { return x > static_cast<Primitive>(0) ? 1 : 0; };

  Vector<Primitive, size> y = x.itor(func);
  return y;
}

template<typename T>
constexpr T step_function(T x)
{
  static_assert(std::is_arithmetic_v<T>);
  return x > 0 ? 1 : 0;
}

template<typename Primitive, std::size_t size>
Vector<Primitive, size> sigmoid(const Vector<Primitive, size> &x)
{
  static_assert(std::is_arithmetic_v<Primitive>);
  const auto one = static_cast<Primitive>(1);
  auto func = [one](Primitive x) -> Primitive
  { return one / (one + exp(-x)); };

  Vector<Primitive, size> y = x.itor(func);
  return y;
}

template<typename T>
constexpr T sigmoid(T x)
{
  static_assert(std::is_arithmetic_v<T>);
  const auto one = static_cast<T>(1);
  return one / (one + exp(-x));
}

template<typename Primitive, std::size_t size>
constexpr
Vector<Primitive, size> relu(const Vector<Primitive, size> &x)
{
  static_assert(std::is_arithmetic_v<Primitive>);
  auto func = [](Primitive x) -> Primitive
  { return x > 0 ? x : 0; };

  Vector<Primitive, size> y = x.itor(func);
  return y;
}

template<typename T>
constexpr
T relu(T x)
{
  static_assert(std::is_arithmetic_v<T>);
  return x > 0 ? x : 0;
}

} // namespace nagato
#endif //DEEPLEARNINGZERO_SRC_ACTIVATION_FUNC_HPP_
