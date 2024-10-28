#include "../include/chessnet.h"
#include <iostream>

ChessNet::ChessNet() :
    conv1(torch::nn::Conv2dOptions(18, 64, 3).stride(1).padding(1)),
    conv2(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
    conv3(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
    conv4(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1)),
    bn1(64), bn2(128), bn3(256), bn4(512),
    conv_flat_bn(512 * 8 * 8),
    fc1(512 * 8 * 8, 1024),
    fc2(1024, 256),
    fc2_bn(256),  // Batch normalization layer for fc2 output
    fc3(256, 1) {

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
    register_module("fc2", fc2);
    register_module("fc2_bn", fc2_bn);  // Register BatchNorm layer
    register_module("fc3", fc3);

    // initialize_weights();
}

torch::Tensor ChessNet::forward(torch::Tensor x)
{
    // x = torch::leaky_relu(bn1(conv1(x)), 0.01);
    // x = torch::leaky_relu(bn2(conv2(x)), 0.01);
    // x = torch::leaky_relu(bn3(conv3(x)), 0.01);
    // x = torch::leaky_relu(bn4(conv4(x)), 0.01);

    x = torch::tanh(bn1(conv1(x)));
    x = torch::tanh(bn2(conv2(x)));
    x = torch::tanh(bn3(conv3(x)));
    x = torch::tanh(bn4(conv4(x)));

    x = x.view({-1, 512 * 8 * 8});

    if (x.size(0) == 1)
    {
        fc2_bn->eval();
        conv_flat_bn->eval();
    }
    else
    {
        fc2_bn->train();
        conv_flat_bn->train();
    }

    x = conv_flat_bn(x);
    // std::cout << "fc1 output range: " << x.min().item().toFloat() << " to " << x.max().item().toFloat() << std::endl;
    x = torch::tanh(fc1(x));
    x = fc2(x);
    x = fc2_bn(x); // Apply BatchNorm after fc2
    // std::cout << "fc2 output range: " << x.min().item().toFloat() << " to " << x.max().item().toFloat() << std::endl;

    x = torch::tanh(fc3(x));
    // std::cout << "output range: " << x.min().item().toFloat() << " to " << x.max().item().toFloat() << std::endl;

    return x;
}

void ChessNet::initialize_weights()
{
    for (auto &module : modules(/*include_self=*/false))
    {
        if (auto *conv = dynamic_cast<torch::nn::Conv2dImpl *>(module.get()))
        {
            // Xavier initialization for Conv2d weights
            torch::nn::init::xavier_uniform_(conv->weight);
            if (conv->options.bias())
            {
                torch::nn::init::zeros_(conv->bias);
            }
        }
        else if (auto *fc = dynamic_cast<torch::nn::LinearImpl *>(module.get()))
        {
            // Xavier initialization for Linear layers
            torch::nn::init::xavier_uniform_(fc->weight);
            if (fc->options.bias())
            {
                torch::nn::init::zeros_(fc->bias);
            }
        }
    }
}