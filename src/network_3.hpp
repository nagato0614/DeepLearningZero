//
// Created by toru on 2023/12/27.
//

#ifndef DEEPLEARNINGZERO_SRC_NETWORK_3_HPP_
#define DEEPLEARNINGZERO_SRC_NETWORK_3_HPP_

#include <iostream>
#include <map>
#include "nagatolib.hpp"

namespace nagato
{

std::map<std::string, MatrixNf> init_network();

MatrixNf forward(const std::map<std::string, MatrixNf> &network,
                 const MatrixNf &x);

MatrixNf predict(const std::map<std::string, MatrixNf> &network,
                 const MatrixNf &x);

/**
 * batch で predict する
 * @param network
 * @param x
 * @return
 */
std::vector<MatrixNf>
predict(const std::map<std::string, MatrixNf> &network,
        const std::vector<MatrixNf> &x);

}

#endif //DEEPLEARNINGZERO_SRC_NETWORK_3_HPP_
