#include <torch/torch.h>
#include <sqlite3.h>
#include <iostream>
#include "../include/chessnet.h"
#include "../include/data_loader.h"



int main() {
    sqlite3* db;
    if (sqlite3_open("/mnt/c/MAKS_STUDIA/MGR/test/chess_evals.db", &db)) {
        std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
        return 1;
    }

    ChessNet net;
    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-4));

    const int batch_size = 1000;
    const int num_epochs = 10;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto data_batch = load_data(db, batch_size, epoch);

        if (data_batch.empty()) {
            std::cerr << "No data found in database." << std::endl;
            break;
        }

        std::vector<torch::Tensor> inputs, targets;

        for (const auto& data : data_batch) {
            auto input_tensor = bitboards_to_tensor(data.bitboards).unsqueeze(0); // Add batch dimension
            inputs.push_back(input_tensor);
            targets.push_back(torch::tensor(data.evaluation, torch::dtype(torch::kFloat32)));
        }

        auto input_tensor = torch::cat(inputs, 0); // (batch_size, 14, 8, 8)
        auto target_tensor = torch::stack(targets);

        optimizer.zero_grad();

        auto output = net.forward(input_tensor);

        auto loss = torch::mse_loss(output.squeeze(), target_tensor);

        loss.backward();
        optimizer.step();

        std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "], Loss: " << loss.item<float>() << std::endl;

        // Overwrite the same file with the latest model weights
    }
    std::string model_path = "../NN_weights/test_model.pt";
    torch::serialize::OutputArchive output_archive;
    // net->save(output_archive);
    net.save(output_archive);
    output_archive.save_to(model_path);
    // torch::save(net, "model_weights.pt");
    // std::ostringstream filename;
    // filename << "weights" << "_loss_" << ".pt";
    // torch::save(net, filename);

    sqlite3_close(db);
    return 0;
}
