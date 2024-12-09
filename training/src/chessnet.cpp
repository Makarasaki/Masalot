#include "../include/chessnet.h"
#include <iostream>

ChessNet::ChessNet() : conv1(torch::nn::Conv2dOptions(18, 64, 3).stride(1).padding(1)),
                       conv2(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
                       conv3(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
                       conv4(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1)),
                       bn1(64), bn2(128), bn3(256), bn4(512),
                       conv_flat_bn(512 * 8 * 8),
                       fc1(512 * 8 * 8, 1024),
                       fc1_bn(1024),
                       fc2(1024, 256),
                       fc2_bn(256),
                       fc3_bn(1),
                       fc3(256, 1)
{

    // Register modules
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("conv4", conv4);
    register_module("bn1", bn1);
    register_module("bn2", bn2);
    register_module("bn3", bn3);
    register_module("bn4", bn4);
    register_module("conv_flat_bn", conv_flat_bn);
    register_module("fc1", fc1);
    register_module("fc1_bn", fc1_bn);
    register_module("fc2", fc2);
    register_module("fc2_bn", fc2_bn);
    register_module("fc3_bn", fc3_bn);
    register_module("fc3", fc3);

    // Initialize weights (if not called elsewhere)
    initialize_weights();
}

torch::Tensor ChessNet::forward(torch::Tensor x)
{
    // Apply convolutional layers with BatchNorm and Tanh activation
    if (x.size(0) == 1)
    {
        // Set specific BatchNorm layers to evaluation mode
        conv_flat_bn->eval();
        fc2_bn->eval();
        fc1_bn->eval();
        fc3_bn->eval();
    }
    else
    {
        // Set specific BatchNorm layers to training mode
        conv_flat_bn->train();
        fc2_bn->train();
        fc1_bn->train();
        fc3_bn->train();
    }

    x = torch::tanh(bn1(conv1(x)));
    x = torch::tanh(bn2(conv2(x)));
    x = torch::tanh(bn3(conv3(x)));
    x = torch::tanh(bn4(conv4(x)));

    // Flatten the output for fully connected layers
    x = x.view({-1, 512 * 8 * 8});
    x = conv_flat_bn(x);

    // Apply first fully connected layer with BatchNorm and Tanh
    x = torch::tanh(fc1_bn(fc1(x)));

    // Apply second fully connected layer with BatchNorm and Tanh
    x = torch::tanh(fc2_bn(fc2(x)));
    // std::cout << "fc2 output range: " << x.min().item().toFloat() << " to " << x.max().item().toFloat() << std::endl;

    // Final layer with Tanh to map output to range [-1, 1]
    x = torch::tanh(fc3_bn(fc3(x)));
    // std::cout << "output avg: " << x.mean().item().toFloat() << std::endl;
    // std::cout << "output range: " << x.min().item().toFloat() << " to " << x.max().item().toFloat() << std::endl;

    return x;
}

torch::Tensor bitboards_to_tensor(const std::vector<std::vector<std::vector<int>>> &bitboards)
{
    std::vector<torch::Tensor> channels;
    for (const auto &board : bitboards)
    {
        torch::Tensor board_tensor = torch::from_blob(const_cast<int *>(board[0].data()), {8, 8}, torch::kInt32).clone();
        channels.push_back(board_tensor.unsqueeze(0));
    }

    return torch::cat(channels, 0).to(torch::kFloat32);
}

void ChessNet::initialize_weights()
{
    for (auto &module : modules(/*include_self=*/false))
    {
        if (auto *conv = dynamic_cast<torch::nn::Conv2dImpl *>(module.get()))
        {
            torch::nn::init::kaiming_uniform_(conv->weight, /*a=*/0.01, torch::kFanIn, torch::kLeakyReLU);
            if (conv->options.bias())
            {
                torch::nn::init::constant_(conv->bias, 0.01);
            }
        }
        else if (auto *fc = dynamic_cast<torch::nn::LinearImpl *>(module.get()))
        {
            torch::nn::init::kaiming_uniform_(fc->weight, /*a=*/0.01, torch::kFanIn, torch::kLeakyReLU);
            if (fc->options.bias())
            {
                torch::nn::init::constant_(fc->bias, 0.01);
            }
        }
        else if (auto *bn1d = dynamic_cast<torch::nn::BatchNorm1dImpl *>(module.get()))
        {
            // Initialize BatchNorm1d weights and biases
            torch::nn::init::ones_(bn1d->weight);
            torch::nn::init::zeros_(bn1d->bias);
        }
        else if (auto *bn2d = dynamic_cast<torch::nn::BatchNorm2dImpl *>(module.get()))
        {
            // Initialize BatchNorm2d weights and biases
            torch::nn::init::ones_(bn2d->weight);
            torch::nn::init::zeros_(bn2d->bias);
        }
    }
}

// void ChessNet::initialize_weights() {
//     for (auto& module : modules(/*include_self=*/false)) {
//         if (auto* conv = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
//             torch::nn::init::kaiming_uniform_(conv->weight, /*a=*/0.01, torch::kFanIn, torch::kLeakyReLU);
//             if (conv->options.bias()) {
//                 // torch::nn::init::zeros_(conv->bias);
//                 torch::nn::init::constant_(conv->bias);
//             }
//         } else if (auto* fc = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
//             torch::nn::init::kaiming_uniform_(fc->weight, /*a=*/0.01, torch::kFanIn, torch::kLeakyReLU);
//             if (fc->options.bias()) {
//                 // torch::nn::init::zeros_(fc->bias);
//                 torch::nn::init::constant_(fc->bias);
//             }
//         }
//     }
// }

// void ChessNet::initialize_weights() {
//     for (auto& module : modules(/*include_self=*/false)) {
//         if (auto* conv = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
//             // Xavier initialization for Conv2d weights
//             torch::nn::init::xavier_uniform_(conv->weight);
//             if (conv->options.bias()) {
//                 torch::nn::init::zeros_(conv->bias);
//                 // torch::nn::init::uniform_(conv->bias, -0.01);
//             }
//         } else if (auto* fc = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
//             // Xavier initialization for Linear layers
//             torch::nn::init::xavier_uniform_(fc->weight);
//             if (fc->options.bias()) {
//                 torch::nn::init::zeros_(fc->bias);
//                 // torch::nn::init::uniform_(conv->bias, -0.01);
//             }
//         }
//     }
// }

// void ChessNet::initialize_weights() {
//     for (auto& module : modules(/*include_self=*/false)) {
//         if (auto* conv = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
//             // Initialize Conv2d weights with very small values (e.g., uniform distribution)
//             torch::nn::init::uniform_(conv->weight, -0.01);
//             if (conv->options.bias()) {
//                 torch::nn::init::zeros_(conv->bias);
//             }
//         } else if (auto* fc = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
//             // Initialize Linear weights with very small values (e.g., uniform distribution)
//             torch::nn::init::uniform_(fc->weight, -0.01);
//             if (fc->options.bias()) {
//                 torch::nn::init::zeros_(fc->bias);
//             }
//         }
//     }
// }
