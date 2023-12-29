//
// Created by toru on 2023/12/28.
//

#ifndef DEEPLEARNINGZERO_SRC_TWO_LAYER_NET_HPP_
#define DEEPLEARNINGZERO_SRC_TWO_LAYER_NET_HPP_

#include "nagatolib.hpp"
#include <map>

namespace nagato
{

class TwoLayerNet
{
 public:
  TwoLayerNet(std::size_t input_size, std::size_t hidden_size,
              std::size_t output_size, float weight_init_std = 0.01);

  /**
   * @brief 予測. batch 処理未対応
   * @param x
   * @return
   */
  MatrixNf predict(const MatrixNf &x) const noexcept;

  /**
   * @brief 予測. batch 処理対応
   * @param x
   * @return
   */
  std::vector<MatrixNf> predict(const std::vector<MatrixNf> &x) const noexcept;

  /**
   * @brief 損失関数. batch 処理未対応
   * @param x
   * @param t
   * @return
   */
  float loss(const MatrixNf &x, const MatrixNf &t) const noexcept;

  /**
   * @brief 損失関数. batch 処理対応
   * @param x
   * @param t
   * @return
   */
  float loss(const std::vector<MatrixNf> &x,
             const std::vector<MatrixNf> &t) const noexcept;

  /**
   * @brief 正解率. batch 処理未対応
   * @param x
   * @param t
   * @return
   */
  float accuracy(const MatrixNf &x, const MatrixNf &t) const noexcept;

  /**
   * @brief 正解率. batch 処理対応
   * @param x
   * @param t
   * @return
   */
  float accuracy(const std::vector<MatrixNf> &x,
                              const std::vector<MatrixNf> &t) const noexcept;

  /**
   * @brief 勾配. batch 処理未対応
   * @param x
   * @param t
   * @return
   */
  std::map<std::string, MatrixNf> numerical_gradient(const MatrixNf &x,
                                                     const MatrixNf &t);

  std::map<std::string, MatrixNf> numerical_gradient(
    const std::vector<MatrixNf> &x,
    const std::vector<MatrixNf> &t);

  std::map<std::string, MatrixNf> params_;
};

}

#endif //DEEPLEARNINGZERO_SRC_TWO_LAYER_NET_HPP_
