//
// Created by toru on 2023/12/28.
//
#include "load_mnist.hpp"
#include "nlohmann/json.hpp"

namespace nagato
{

std::pair<std::map<std::string, std::vector<std::vector<float>>>,
          std::map<std::string, std::vector<float>>>
load_mnist(const std::string &json_file_path)
{
  using json = nlohmann::json;

  // JSONファイルを開く
  std::ifstream json_file(json_file_path);
  if (!json_file.is_open())
  {
    std::cerr << "Failed to open " << json_file_path << std::endl;
    throw std::runtime_error("Failed to open " + json_file_path);
  }

  // JSONデータを読み込む
  json json_data;
  json_file >> json_data;

  // JSONデータから必要な情報を取得する
  std::vector<std::vector<float >> train_imgs =
    json_data["train_img"].get<std::vector<std::vector<float>>>();
  std::cout << "Number of training images: " << train_imgs.size() << std::endl;

  std::vector<float> train_labels =
    json_data["train_label"].get<std::vector<float>>();
  std::cout << "Number of training labels: " << train_labels.size() << std::endl;

  std::vector<std::vector<float >>
    test_imgs =
    json_data["test_img"].get<std::vector<std::vector<float>>>();
  std::cout << "Number of test images: " << test_imgs.size() << std::endl;

  std::vector<float> test_labels =
    json_data["test_label"].get<std::vector<float>>();
  std::cout << "Number of test labels: " << test_labels.size() << std::endl;

  std::map<std::string, std::vector<std::vector<float>>> img_data;
  img_data.insert(std::make_pair("train_img", train_imgs));
  img_data.insert(std::make_pair("test_img", test_imgs));

  std::map<std::string, std::vector<float>> label_data;
  label_data.insert(std::make_pair("train_label", train_labels));
  label_data.insert(std::make_pair("test_label", test_labels));

  return std::make_pair(img_data, label_data);
}

void show_img(const std::vector<float> &img, const std::string &filename)
{
  std::cout << "Saving " << filename << std::endl;

  // ファイルを開く
  std::ofstream ofs(filename);

  // ヘッダー
  ofs << "P3" << std::endl;
  ofs << "28 28" << std::endl;
  ofs << "255" << std::endl;

  // データ
  for (auto i = 0uz; i < img.size(); i++)
  {
    const auto pixel = static_cast<int>(img[i] * 255);
    ofs << pixel << " " << pixel << " " << pixel << std::endl;
  }

  // ファイルを閉じる
  ofs.close();
}

std::pair<std::map<std::string, std::vector<std::vector<float>>>,
          std::map<std::string, std::vector<float>>>
load_weight(const std::string &json_file_path)
{
  using json = nlohmann::json;

  // JSONファイルを開く
  std::ifstream json_file(json_file_path);
  if (!json_file.is_open())
  {
    std::cerr << "Failed to open " << json_file_path << std::endl;
    throw std::runtime_error("Failed to open " + json_file_path);
  }

  // JSONデータを読み込む
  json json_data;
  json_file >> json_data;

  // JSONデータから必要な情報を取得する
  std::vector<std::vector<float >> w1 =
    json_data["W1"].get<std::vector<std::vector<float>>>();
  std::cout << "Number of w1: " << w1.size() << std::endl;
  std::cout << "Number of w1[0]: " << w1[0].size() << std::endl;

  std::vector<std::vector<float >> w2 =
    json_data["W2"].get<std::vector<std::vector<float>>>();
  std::cout << "Number of w2: " << w2.size() << std::endl;
  std::cout << "Number of w2[0]: " << w2[0].size() << std::endl;

  std::vector<std::vector<float >> w3 =
    json_data["W3"].get<std::vector<std::vector<float>>>();
  std::cout << "Number of w3: " << w3.size() << std::endl;
  std::cout << "Number of w3[0]: " << w3[0].size() << std::endl;

  std::vector<float> b1 =
    json_data["b1"].get<std::vector<float>>();
  std::cout << "Number of b1: " << b1.size() << std::endl;

  std::vector<float> b2 =
    json_data["b2"].get<std::vector<float>>();
  std::cout << "Number of b2: " << b2.size() << std::endl;

  std::vector<float> b3 =
    json_data["b3"].get<std::vector<float>>();
  std::cout << "Number of b3: " << b3.size() << std::endl;

  std::map<std::string, std::vector<std::vector<float>>> weight;
  weight.insert(std::make_pair("W1", w1));
  weight.insert(std::make_pair("W2", w2));
  weight.insert(std::make_pair("W3", w3));

  std::map<std::string, std::vector<float>> bias;
  bias.insert(std::make_pair("b1", b1));
  bias.insert(std::make_pair("b2", b2));
  bias.insert(std::make_pair("b3", b3));
  return std::make_pair(weight, bias);
}

}