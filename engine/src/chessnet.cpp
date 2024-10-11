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

void ChessNet::initialize_weights() {
    for (auto& module : modules(/*include_self=*/false)) {
        if (auto* conv = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
            // Xavier initialization for Conv2d weights
            torch::nn::init::xavier_uniform_(conv->weight);
            if (conv->options.bias()) {  // Check if bias exists for the conv layer
                torch::nn::init::zeros_(conv->bias);  // No dereference needed, pass bias tensor directly
            }
        } else if (auto* fc = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
            // Xavier initialization for Linear layers
            torch::nn::init::xavier_uniform_(fc->weight);
            if (fc->options.bias()) {  // Check if bias exists for the linear layer
                torch::nn::init::zeros_(fc->bias);  // No dereference needed, pass bias tensor directly
            }
        }
    }
}