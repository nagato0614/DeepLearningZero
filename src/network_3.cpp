//
// Created by toru on 2023/12/27.
//
#include "network_3.hpp"
#include "activation_func.hpp"
#include "load_mnist.hpp"

namespace nagato
{

std::map<std::string, MatrixNf> init_network()
{
  std::map<std::string, MatrixNf> network;

  const auto [weight, bias] =
    load_weight("../dataset/sample_weight.json");

  const auto W1 = MatrixNf(weight.at("W1"));
  const auto W2 = MatrixNf(weight.at("W2"));
  const auto W3 = MatrixNf(weight.at("W3"));

  const auto b1 = MatrixNf(bias.at("b1"));
  const auto b2 = MatrixNf(bias.at("b2"));
  const auto b3 = MatrixNf(bias.at("b3"));

  network.insert(std::make_pair("W1", W1));
  network.insert(std::make_pair("W2", W2));
  network.insert(std::make_pair("W3", W3));
  network.insert(std::make_pair("b1", b1));
  network.insert(std::make_pair("b2", b2));
  network.insert(std::make_pair("b3", b3));

  return network;
}

MatrixNf forward(const std::map<std::string, MatrixNf> &network,
                                 const nagato::MatrixNf &x)
{
  const auto &W1 = network.at("W1");
  const auto &W2 = network.at("W2");
  const auto &W3 = network.at("W3");
  const auto &b1 = network.at("b1");
  const auto &b2 = network.at("b2");
  const auto &b3 = network.at("b3");

  const auto a1 = Dot(x, W1) + b1;
  const auto z1 = sigmoid(a1);
  const auto a2 = Dot(z1, W2) + b2;
  const auto z2 = sigmoid(a2);
  const auto a3 = Dot(z2, W3) + b3;
  const auto y = identity_function(a3);

  return y;
}

MatrixNf predict(const std::map<std::string, MatrixNf> &network,
                         const MatrixNf &x)
{
  const auto &W1 = network.at("W1");
  const auto &W2 = network.at("W2");
  const auto &W3 = network.at("W3");
  const auto &b1 = network.at("b1");
  const auto &b2 = network.at("b2");
  const auto &b3 = network.at("b3");

  const auto a1 = Dot(x.ToVector(), W1) + b1;
  const auto z1 = sigmoid(a1);
  const auto a2 = Dot(z1, W2) + b2;
  const auto z2 = sigmoid(a2);
  const auto a3 = Dot(z2, W3) + b3;
  const auto y = softmax(a3);

  return y;
}

std::vector<MatrixNf> predict(const std::map<std::string, MatrixNf> &network,
                              const std::vector<MatrixNf> &x)
{
  std::vector<MatrixNf> y;
  for (const auto &x_i : x)
  {
    y.push_back(predict(network, x_i));
  }
  return y;
}
}

