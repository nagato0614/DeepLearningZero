//
// Created by toru on 2023/12/28.
//

#include "two_layer_net.hpp"
#include "activation_func.hpp"
#include "loss_functions.hpp"
#include "nagatolib.hpp"
#include "numerical.hpp"
#include <iostream>

namespace nagato
{

nagato::TwoLayerNet::TwoLayerNet(std::size_t input_size,
                                 std::size_t hidden_size,
                                 std::size_t output_size,
                                 float weight_init_std)
{
  const auto W1 = weight_init_std * Randn<float>(input_size, hidden_size);
  const auto b1 = MatrixNf::Zero(1, hidden_size);
  const auto W2 = weight_init_std * Randn<float>(hidden_size, output_size);
  const auto b2 = MatrixNf::Zero(1, output_size);

  this->params_.insert(std::make_pair("W1", W1));
  this->params_.insert(std::make_pair("b1", b1));
  this->params_.insert(std::make_pair("W2", W2));
  this->params_.insert(std::make_pair("b2", b2));
}

MatrixNf TwoLayerNet::predict(const MatrixNf &x) const noexcept
{
  const auto &W1 = this->params_.at("W1");
  const auto &W2 = this->params_.at("W2");
  const auto &b1 = this->params_.at("b1");
  const auto &b2 = this->params_.at("b2");

  const auto a1 = Dot(x, W1) + b1;
  const auto z1 = sigmoid(a1);
  const auto a2 = Dot(z1, W2) + b2;
  const auto y = Softmax(a2);

  return y;
}

std::vector<MatrixNf> TwoLayerNet::predict(const std::vector<MatrixNf> &x) const noexcept
{
  std::vector<MatrixNf> y(x.size());
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (auto i = 0uz; i < x.size(); i++)
  {
    y.at(i) = predict(x.at(i));
  }
  return y;
}

float TwoLayerNet::loss(const MatrixNf &x, const MatrixNf &t) const noexcept
{
  const auto y = predict(x);
  const auto loss = CrossEntropyError(y, t);
  return loss;
}

float TwoLayerNet::loss(const std::vector<MatrixNf> &x,
                        const std::vector<MatrixNf> &t) const noexcept
{
  const auto y = predict(x);
  float loss = CrossEntropyError(y, t);
  return loss;
}

float TwoLayerNet::accuracy(const MatrixNf &x, const MatrixNf &t) const noexcept
{
  const auto y = predict(x);
  const auto [y_row, y_column] = ArgMax(y);
  const auto [t_row, t_column] = ArgMax(t);

  const auto accuracy = y_column == t_column ? 1.0f : 0.0f;

  return accuracy;
}

float TwoLayerNet::accuracy(const std::vector<MatrixNf> &x,
                            const std::vector<MatrixNf> &t) const noexcept
{
  const auto y = predict(x);
  float accuracy = 0.0f;
  for (auto i = 0uz; i < y.size(); i++)
  {
    const auto [y_row, y_column] = ArgMax(y.at(i));
    const auto [t_row, t_column] = ArgMax(t.at(i));

    accuracy += y_column == t_column ? 1.0f : 0.0f;
  }
  return accuracy / y.size();
}

std::map<std::string, MatrixNf>
TwoLayerNet::numerical_gradient(const MatrixNf &x,
                                const MatrixNf &t)
{
  const auto loss_W = [&](const MatrixNf &W)
  {
    return loss(x, t);
  };

  std::map<std::string, MatrixNf> grads;
  const std::vector<std::string> param_list =
    {
      "W1",
      "b1",
      "W2",
      "b2"
    };

  for (auto i = 0uz; i < param_list.size(); i++)
  {
    const auto param = param_list.at(i);
    grads.at(param) = gradient(loss_W, this->params_.at(param));
  };

  return grads;

}

std::map<std::string, MatrixNf>
TwoLayerNet::numerical_gradient(
  const std::vector<MatrixNf> &x,
  const std::vector<MatrixNf> &t)
{
  const auto loss_W = [&](const MatrixNf &W)
  {
    return loss(x, t);
  };

  std::map<std::string, MatrixNf> grads;
  grads.insert(std::make_pair("W1", gradient(loss_W, this->params_.at("W1"))));
  grads.insert(std::make_pair("b1", gradient(loss_W, this->params_.at("b1"))));
  grads.insert(std::make_pair("W2", gradient(loss_W, this->params_.at("W2"))));
  grads.insert(std::make_pair("b2", gradient(loss_W, this->params_.at("b2"))));

  return grads;
}

}