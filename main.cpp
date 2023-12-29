#include <iostream>
#include <fstream>
#include <omp.h>
#include "nagatolib.hpp"

#include "network_3.hpp"
#include "load_mnist.hpp"
#include "two_layer_net.hpp"
#include "relu.hpp"

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

void batch_prediction()
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
}

void two_layer_net()
{
  using namespace nagato;

  // openMPが有効な場合は、スレッド数を表示する
  // openMPが有効な場合は、スレッド数を表示する
#ifdef _OPENMP
  std::cout << "Number of threads: " << omp_get_num_threads() << std::endl;
#endif

  TwoLayerNet network(784, 100, 10);
  network.params_.at("W1").ShowShape();
  network.params_.at("b1").ShowShape();
  network.params_.at("W2").ShowShape();
  network.params_.at("b2").ShowShape();

  const auto [img_data, label_data] = load_mnist("../dataset/mnist.json");

  const auto train_img = img_data.at("train_img");
  const auto train_label = label_data.at("train_label");
  const auto test_img = img_data.at("test_img");
  const auto test_label = label_data.at("test_label");

  // テスト画像を MatrixNf に変換
  const auto test_img_matrix = convert_img_to_matrix(test_img);

  std::vector<float> train_loss_list;
  std::vector<float> train_acc_list;
  std::vector<float> test_acc_list;

  const std::vector<std::string> param_list =
    {
      "W1",
      "b1",
      "W2",
      "b2"
    };

  Random random;

  // ハイパーパラメータ
  const auto batch_size = 1000uz;
  const auto iter_per_epoch =
    std::max(static_cast<std::size_t>(train_img.size() / batch_size), 1uz);
  constexpr auto learning_rate = 0.1f;
  const auto iter_num = 10000uz;

  for (auto i = 0uz; i < iter_num; i++)
  {
    const auto [train_img_batch, train_label_batch] =
      choice_mini_batch(train_img, train_label, batch_size);

    const auto label_on_hot = convert_one_hot(train_label_batch, 10);

    const auto grad = network.numerical_gradient(train_img_batch, label_on_hot);

    for (const auto &param : param_list)
    {
      network.params_.at(param) -= learning_rate * grad.at(param);
    }

    const auto loss = network.loss(train_img_batch, label_on_hot);
    std::cout << "loss: " << loss << std::endl;
    train_loss_list.emplace_back(loss);

    // 1エポックごとに認識精度を計算
    if (i % iter_per_epoch == 0)
    {
      // 訓練データでの認識精度を計算
      const auto [t_img, t_label] =
        choice_mini_batch(train_img, train_label, 10);
      const auto l_on_hot = convert_one_hot(t_label, 10);
      const auto train_acc = network.accuracy(
        t_img,
        l_on_hot
      );

      // テストデータでの認識精度を計算
      const auto test_label_on_hot = convert_one_hot(test_label, 10);
      const auto test_acc = network.accuracy(
        test_img_matrix,
        test_label_on_hot
      );

      std::cout << "train acc, test acc | " << train_acc << ", " << test_acc << std::endl;
      train_acc_list.emplace_back(train_acc);
      test_acc_list.emplace_back(test_acc);
    }
  }
}

int main()
{
  using namespace nagato;

  Relu relu;
  MatrixNf x{
    {1.0f, -0.5f},
    {-2.0f, 3.0f}
  };
  std::cout << x << std::endl;
  std::cout << relu.Forward(x) << std::endl;
  std::cout << relu.mask_ << std::endl;
  std::cout << relu.Backward(x) << std::endl;

  return 0;
}