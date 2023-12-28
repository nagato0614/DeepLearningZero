//
// Created by toru on 2023/12/28.
//

#ifndef DEEPLEARNINGZERO_SRC_LOAD_MNIST_HPP_
#define DEEPLEARNINGZERO_SRC_LOAD_MNIST_HPP_

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

namespace nagato
{
std::pair<std::map<std::string, std::vector<std::vector<float>>>,
          std::map<std::string, std::vector<float>>>
load_mnist(const std::string &json_file_path);

/**
 * ppm 画像として白黒画像を保存する
 * @param img
 */
void show_img(const std::vector<float> &img,
              const std::string &filename);

/**
 * weightを取得する
 */
std::pair<std::map<std::string, std::vector<std::vector<float>>>,
          std::map<std::string, std::vector<float>>>
load_weight(const std::string &json_file_path);

}

#endif //DEEPLEARNINGZERO_SRC_LOAD_MNIST_HPP_
