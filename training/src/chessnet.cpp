#include "../include/chessnet.h"
#include <iostream>

ChessNet::ChessNet() :
    conv1(torch::nn::Conv2dOptions(14, 64, 3).stride(1).padding(1)),
    conv2(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
    conv3(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
    conv4(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1)),
    bn1(64), bn2(128), bn3(256), bn4(512),
    fc1(512 * 8 * 8, 1024),
    fc2(1024, 256),
    fc3(256, 1) {

    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("conv4", conv4);
    register_module("bn1", bn1);
    register_module("bn2", bn2);
    register_module("bn3", bn3);
    register_module("bn4", bn4);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
}

torch::Tensor ChessNet::forward(torch::Tensor x) {
    x = torch::relu(bn1(conv1(x)));
    x = torch::relu(bn2(conv2(x)));
    x = torch::relu(bn3(conv3(x)));
    x = torch::relu(bn4(conv4(x)));

    x = x.view({-1, 512 * 8 * 8});

    x = torch::relu(fc1(x));
    x = torch::relu(fc2(x));
    x = torch::tanh(fc3(x));

    return x;
}

// void ChessNet::save(torch::serialize::OutputArchive& archive) const {
//     archive.write("conv1", conv1);
//     archive.write("conv2", conv2);
//     archive.write("fc1", fc1);
//     archive.write("fc2", fc2);
// }

// void ChessNet::load(torch::serialize::InputArchive& archive) {
//     archive.read("conv1", conv1);
//     archive.read("conv2", conv2);
//     archive.read("fc1", fc1);
//     archive.read("fc2", fc2);
// }

torch::Tensor bitboards_to_tensor(const std::vector<std::vector<std::vector<int>>>& bitboards) {
    std::vector<torch::Tensor> channels;

    for (const auto& board : bitboards) {
        torch::Tensor board_tensor = torch::from_blob(const_cast<int*>(board[0].data()), {8, 8}, torch::kInt32).clone();
        channels.push_back(board_tensor.unsqueeze(0));
    }

    return torch::cat(channels, 0).to(torch::kFloat32);
}
