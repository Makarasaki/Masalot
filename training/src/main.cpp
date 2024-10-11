#include <torch/torch.h>
#include <sqlite3.h>
#include <iostream>
#include "../include/chessnet.h"
#include "../include/data_loader.h"

void save_model(ChessNet &net, const torch::Device &device, const std::string &path)
{
    torch::serialize::OutputArchive output_archive;
    net.to(torch::kCPU); // Move model to CPU for saving
    net.save(output_archive);
    output_archive.save_to(path);
    net.to(device); // Move back to the device after saving
    std::cout << "Model saved to " << path << std::endl;
}

int main()
{
    // Check if CUDA is available
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available())
    {
        device = torch::Device(torch::kCUDA);
        std::cout << "CUDA is available! Using GPU." << std::endl;
    }
    else
    {
        std::cout << "CUDA is not available. Using CPU." << std::endl;
    }

    sqlite3 *db;
    if (sqlite3_open("../../data/chess_evals.db", &db))
    {
        std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
        return 1;
    }

    // Instantiate model and move it to the chosen device
    ChessNet net;
    net.to(device);

    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-4));

    const int batch_size = 5000;
    const int num_epochs = 2400;

    float min_loss = std::numeric_limits<float>::max(); // Track the minimum loss
    std::string model_path = "../NN_weights/best_model.pt";

    for (int epoch = 0; epoch < num_epochs; ++epoch)
    {
        auto data_batch = load_data(db, batch_size, epoch);

        if (data_batch.empty())
        {
            std::cerr << "No data found in database." << std::endl;
            break;
        }

        std::vector<torch::Tensor> inputs, targets;

        for (const auto &data : data_batch)
        {
            auto input_tensor = bitboards_to_tensor(data.bitboards).unsqueeze(0).to(device); // Add batch dimension and move to device
            inputs.push_back(input_tensor);
            targets.push_back(torch::tensor(data.evaluation, torch::dtype(torch::kFloat32)).to(device)); // Move targets to device
        }

        auto input_tensor = torch::cat(inputs, 0); // (batch_size, 14, 8, 8)
        auto target_tensor = torch::stack(targets);

        optimizer.zero_grad();

        auto output = net.forward(input_tensor);

        auto loss = torch::mse_loss(output.squeeze(), target_tensor);

        loss.backward();
        optimizer.step();

        float current_loss = loss.item<float>();
        std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "], Loss: " << current_loss << std::endl;

        // Save the model if the current loss is lower than the minimum loss
        if (current_loss < min_loss)
        {
            min_loss = current_loss;
            save_model(net, device, model_path);
            std::cout << "New minimum loss achieved. Model saved!" << std::endl;
        }

        // Save the model at regular intervals
        if (epoch % 450 == 0)
        {
            std::string epoch_model_path = "../NN_weights/model_epoch_" + std::to_string(epoch) + ".pt";
            save_model(net, device, epoch_model_path);
            std::cout << "Epoch " << epoch << ": Model saved with loss " << current_loss << "!" << std::endl;
        }
    }

    sqlite3_close(db);
    return 0;
}
