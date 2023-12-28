#include <iostream>
#include <fstream>
#include "nagatolib.hpp"

#include "network_3.hpp"
#include "load_mnist.hpp"

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
  using namespace nagato;

  const auto [img_data, label_data] = load_mnist("../dataset/mnist.json");

  const auto network = init_network();
  const auto test_data_size = img_data.at("test_img").size();

  // 正解率
  auto accuracy_cnt = 0uz;
  for (auto i = 0uz; i < test_data_size; i++)
  {
    const auto &test_img = MatrixNf(img_data.at("test_img")[i], 28, 28);
    const auto &test_label = label_data.at("test_label")[i];
    const auto y = predict(network, test_img);
    const auto [row, col] = y.ArgMax();
    std::cout << "label: " << test_label << ", predict: " << col << std::endl;

    if (test_label == col)
    {
      accuracy_cnt++;
    }
  }

  std::cout << "Accuracy: " << static_cast<double>(accuracy_cnt) / test_data_size << std::endl;

  return 0;
}
