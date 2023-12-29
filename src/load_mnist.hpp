//
// Created by toru on 2023/12/28.
//

#ifndef DEEPLEARNINGZERO_SRC_LOAD_MNIST_HPP_
#define DEEPLEARNINGZERO_SRC_LOAD_MNIST_HPP_

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include "nagatolib.hpp"

namespace nagato
{
std::pair<std::map<std::string, std::vector<std::vector<float>>>,
          std::map<std::string, std::vector<float>>>
load_mnist(const std::string &json_file_path);

std::vector<MatrixNf> convert_img_to_matrix(const std::vector<std::vector<float>> &img);

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

/**
 * ミニバッチを取得する
 * @param x
 * @param y
 * @param batch_size
 * @return
 */
std::pair<std::vector<MatrixNf>,
          std::vector<float>>
choice_mini_batch(const std::vector<std::vector<float>> &x,
                  const std::vector<float> &y,
                  std::size_t batch_size);

MatrixNf convert_one_hot(int label, int size);
std::vector<MatrixNf> convert_one_hot(const std::vector<float> &label, int size);

}

#endif //DEEPLEARNINGZERO_SRC_LOAD_MNIST_HPP_
