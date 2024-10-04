#ifndef CHESSNET_H
#define CHESSNET_H

#include <torch/torch.h>
#include <sqlite3.h>
#include <vector>
#include <cstdint>

// struct ChessNet : torch::nn::Module {
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

//     ChessNet();
//     torch::Tensor forward(torch::Tensor x);
// };


// struct ChessNet : torch::nn::Module {
//     torch::nn::Conv2d conv1, conv2;
//     torch::nn::Linear fc1, fc2;

//     ChessNet();  // Constructor declaration

//     torch::Tensor forward(torch::Tensor x);

//     // void save(torch::serialize::OutputArchive& archive) const override;
//     // void load(torch::serialize::InputArchive& archive) override;
// };

struct ChessNet : torch::nn::Module {
    torch::nn::Conv2d conv1, conv2, conv3, conv4;

    torch::nn::BatchNorm2d bn1, bn2, bn3, bn4;

    torch::nn::Linear fc1, fc2, fc3;

    ChessNet();

    torch::Tensor forward(torch::Tensor x);

    // Save and load methods (optional)
    // void save(torch::serialize::OutputArchive& archive) const override;
    // void load(torch::serialize::InputArchive& archive) override;
};


torch::Tensor bitboards_to_tensor(const std::vector<std::vector<std::vector<int>>>& bitboards);

#endif // CHESSNET_H
