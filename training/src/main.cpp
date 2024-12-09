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
    // torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    sqlite3 *db;
    if (sqlite3_open("../../data/chess_evals.db", &db))
    {
        std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
        return 1;
    }

    // Instantiate model and move it to the chosen device
    ChessNet net;
    // std::cout<<"0"<<std::endl;
    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
    // torch::optim::SGD optimizer(net.parameters(), torch::optim::SGDOptions(1e-1).momentum(0.9).weight_decay(1e-4));
    // torch::optim::StepLR scheduler(optimizer, /*step_size=*/6000, /*gamma=*/0.1);

    const int batch_size = 512;
    const int num_epochs = 23437;
    const float alpha = 0.1; // Hyperparameter to encourage range utilization
    float min_loss = std::numeric_limits<float>::max(); // Track the minimum loss
    std::string model_path = "../NN_weights/best_model.pt";


    net.to(device);
    for (int epoch = 0; epoch < num_epochs; ++epoch)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        auto data_batch = load_data(db, batch_size, epoch);
        // std::cout<<"1"<<std::endl;
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
        // std::cout<<"2"<<std::endl;
        auto input_tensor = torch::cat(inputs, 0); // (batch_size, 14, 8, 8), (batch_size, 18, 8, 8)
        auto target_tensor = torch::stack(targets);

        std::cout << "Target avg: " << target_tensor.mean().item().toFloat() << std::endl;
        std::cout << "Target range: " << target_tensor.min().item().toFloat() << " to " << target_tensor.max().item().toFloat() << std::endl;

        optimizer.zero_grad();

        auto output = net.forward(input_tensor);

        // auto loss = torch::l1_loss(output.squeeze(), target_tensor);
        // auto loss = torch::mse_loss(output.squeeze(), target_tensor);
        auto mse_loss = torch::mse_loss(output.squeeze(), target_tensor);
        auto range_penalty = torch::mean(torch::abs(output)); // Encourages output range utilization
        auto loss = mse_loss + alpha * range_penalty;

        loss.backward();
        // torch::nn::utils::clip_grad_norm_(net.parameters(), 1.0);
        optimizer.step();
        // scheduler.step();

        float current_loss = loss.item<float>();

        // Save the model if the current loss is lower than the minimum loss
        if (current_loss < min_loss)
        {
            min_loss = current_loss;
            save_model(net, device, model_path);
            std::cout << "New minimum loss achieved. Model saved!" << std::endl;
        }

        // Save the model at regular intervals
        if (epoch % 6000 == 0)
        {
            std::string epoch_model_path = "../NN_weights/model_epoch_" + std::to_string(epoch) + ".pt";
            save_model(net, device, epoch_model_path);
            std::cout << "Epoch " << epoch << ": Model saved with loss " << current_loss << "!" << std::endl;
        }

        // Record end time and calculate duration
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end_time - start_time;
        std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "], Loss: " << current_loss << std::endl;
        std::cout.flush();
        std::cout << "Time taken for epoch " << epoch + 1 << ": " << duration.count() << " seconds." << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
    }

    std::string epoch_model_path = "../NN_weights/model_last.pt";
    save_model(net, device, epoch_model_path);

    sqlite3_close(db);
    return 0;
}
