#include <torch/torch.h>
#include <sqlite3.h>
#include <iostream>
// #include <cstdint>
#include "../include/chessnet.h"
#include "../include/data_loader.h"

// void save_model(ChessNet &net, const torch::Device &device, const std::string &path)
void save_model(ChessNet &net, const torch::Device &device, const std::string &path)
{
    torch::serialize::OutputArchive output_archive;
    net->to(torch::kCPU); // Move model to CPU for saving
    net->save(output_archive);
    output_archive.save_to(path);
    net->to(device); // Move back to the device after saving
    std::cout << "Model saved to " << path << std::endl;
}

int main()
{
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

    // ChessNet net;
    auto net = ChessNet();
    // std::cout<<"0"<<std::endl;
    torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(1e-5));
    // torch::optim::SGD optimizer(net.parameters(), torch::optim::SGDOptions(1e-1).momentum(0.9).weight_decay(1e-4));
    // torch::optim::StepLR scheduler(optimizer, /*step_size=*/6000, /*gamma=*/0.1);

    const int num_epochs = 10;
    const int batch_size = 512;
    const int num_batches = 12641;
    const float alpha = 0.1;                            // Hyperparameter to use whole rang3e
    float min_loss = std::numeric_limits<float>::max(); // Track the minimum loss
    std::string model_path = "../NN_weights/best_model.pt";

    net->to(device);

    // torch::serialize::InputArchive input_archive;
    // try
    // {
    //     input_archive.load_from("../../training/NN_weights/model_last_w_5e.pt");
    //     net->load(input_archive); // Load the weights into the model
    //     // Check if CUDA is available
    //     if (torch::cuda::is_available())
    //     {
    //         net->to(torch::kCUDA);
    //         std::cout << "Using CUDA" << std::endl;
    //     }
    //     else
    //     {
    //         // If not available, keep on CPU
    //         net->to(torch::kCPU);
    //     }
    //     std::cout << "Model weights loaded successfully!" << std::endl;
    // }
    // catch (const c10::Error &e)
    // {
    //     std::cerr << "Error loading model weights: " << e.what() << std::endl;
    // }

    for (int epoch = 0; epoch < num_epochs; epoch++)
    {
        for (int batch = 0; batch < num_batches; ++batch)
        {
            auto start_time = std::chrono::high_resolution_clock::now();

            BatchData batch_data = load_data(db, batch_size, batch, net, device);

            // If no data was returned, break out of the loop
            if (batch_data.inputs.size(0) == 0) {
                std::cerr << "No data found in database or no valid rows." << std::endl;
                break;
            }

            std::cout << "Input tensor shape: " << batch_data.inputs.sizes() << std::endl;
            std::cout << "Target avg: " << batch_data.targets.mean().item().toFloat() << std::endl;
            std::cout << "Target range: "
                    << batch_data.targets.min().item().toFloat()
                    << " to "
                    << batch_data.targets.max().item().toFloat()
                    << std::endl;

            // Zero the gradients
            optimizer.zero_grad();

            // Forward pass
            std::cout << "idzie do forwarda" << std::endl;
            auto output = net->forward(batch_data.inputs);

            // Compute your loss
            auto mse_loss = torch::mse_loss(output.squeeze(), batch_data.targets);
            auto range_penalty = torch::mean(torch::abs(output));  // optional

            // Option A: pure MSE loss
            mse_loss.backward();

            // Option B: MSE + range penalty
            // float alpha = 0.1f;  // or whichever hyperparameter
            // auto loss = mse_loss + alpha * range_penalty;
            // loss.backward();

            // Update weights
            optimizer.step();

            // If you have a scheduler, step it here (optional)
            // scheduler.step();

            float current_loss = mse_loss.item<float>();

            if (current_loss < min_loss)
            {
                min_loss = current_loss;
                save_model(net, device, model_path);
                std::cout << "New minimum loss achieved. Model saved!" << std::endl;
            }

            if (batch % 6000 == 0)
            {
                std::string epoch_model_path = "../NN_weights/model_epoch_" + std::to_string(batch) + "_" + std::to_string(epoch) + ".pt";
                save_model(net, device, epoch_model_path);
                std::cout << "Epoch " << batch << ": Model saved with loss " << current_loss << "!" << std::endl;
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> duration = end_time - start_time;
            std::cout << "Epoch [" << epoch + 1 << " " << batch + 1 << "/" << num_batches << "], Loss: " << current_loss << std::endl;
            std::cout.flush();
            std::cout << "Time taken for batch " << batch + 1 << ": " << duration.count() << " seconds." << std::endl;
            std::cout << "------------------------------------------------" << std::endl;
        }
    }

    std::string epoch_model_path = "../NN_weights/model_last.pt";
    save_model(net, device, epoch_model_path);

    sqlite3_close(db);
    return 0;
}
