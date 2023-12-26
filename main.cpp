#include <iostream>
#include <fstream>
#include "nagatolib.hpp"

#include "activation_func.hpp"

/**
 * ベクトルをcsvに保存する
 * @return
 */
template<typename Primitive, std::size_t size>
void save_vec(const nagato::Vector<Primitive, size> &x,
              const nagato::Vector<Primitive, size> &y,
              const std::string &filename)
{
  std::ofstream ofs(filename);
  // ヘッダー
  ofs << "x,y" << std::endl;
  for (auto i = 0uz; i < size; i++)
  {
    ofs << x[i] << "," << y[i] << std::endl;
  }
  ofs.close();
}

int main()
{
  std::cout << "step function" << std::endl;
  std::cout << nagato::step_function(0.5) << std::endl;
  std::cout << nagato::step_function(-0.5) << std::endl;

  constexpr auto min = -5.0f;
  constexpr auto max = 5.0f;
  constexpr auto step = 0.1f;
  constexpr auto size = static_cast<std::size_t>((max - min) / step);
  constexpr auto x = nagato::LineSpace<float, size>(min, max);

  const auto sigmoid = nagato::sigmoid(x);
  const auto relu = nagato::relu(x);

  save_vec(x, nagato::step_function(x), "step_function.csv");
  save_vec(x, sigmoid, "sigmoid.csv");
  save_vec(x, relu, "relu.csv");
  return 0;
}
