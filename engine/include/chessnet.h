#ifndef CHESSNET_H
#define CHESSNET_H

#include <torch/torch.h>
#include <sqlite3.h>
#include <vector>
#include <cstdint>

struct ChessNet : torch::nn::Module {
    torch::nn::Conv2d conv1, conv2, conv3, conv4;

    torch::nn::BatchNorm2d bn1, bn2, bn3, bn4;

    torch::nn::Linear fc1, fc2, fc3;

    ChessNet();

    torch::Tensor forward(torch::Tensor x);

};

#endif // CHESSNET_H
