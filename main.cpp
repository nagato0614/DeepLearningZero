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
  const auto batch_size = 100uz;
  for (auto i = 0uz; i < test_data_size; i += batch_size)
  {
    const std::vector<MatrixNf> test_img_batch(
      img_data.at("test_img").begin() + i,
      img_data.at("test_img").begin() + i + batch_size
    );

    const std::vector<float> test_label_batch(
      label_data.at("test_label").begin() + i,
      label_data.at("test_label").begin() + i + batch_size
    );

    const auto y = predict(network, test_img_batch);
    for (auto j = 0uz; j < batch_size; j++)
    {
      const auto [row, col] = y[j].ArgMax();
      std::cout << "label: " << test_label_batch[row] << ", predict: " << col << std::endl;

      if (test_label_batch[j] == col)
      {
        accuracy_cnt++;
      }
    }

  }
  std::cout << "Correct: " << accuracy_cnt << std::endl; // "Correct: 9352
  std::cout << "Accuracy: " << static_cast<double>(accuracy_cnt) / test_data_size << std::endl;

  return 0;
}
