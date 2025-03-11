#include <torch/torch.h>
#include <sqlite3.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include "../../training/include/chessnet.h"
#include "../../training/include/data_loader.h"

// This function loads a trained model from disk into 'net'
bool load_model(ChessNet &net, const std::string &model_path, torch::Device device)
{
    try
    {
        torch::serialize::InputArchive input_archive;
        input_archive.load_from(model_path);
        net->load(input_archive);
        net->to(device);
        net->eval(); // Set model to eval mode for inference
        std::cout << "Model weights loaded successfully from: " << model_path << std::endl;
        return true;
    }
    catch (const c10::Error &e)
    {
        std::cerr << "Error loading model weights: " << e.what() << std::endl;
        return false;
    }
}

int main()
{
    // 1) Choose device
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available())
    {
        device = torch::kCUDA;
        std::cout << "CUDA is available! Using GPU." << std::endl;
    }
    else
    {
        std::cout << "CUDA is not available. Using CPU." << std::endl;
    }

    // 2) Open the database
    sqlite3 *db;
    if (sqlite3_open("../../data/chess_evals.db", &db))
    {
        std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
        return 1;
    }

    // 3) Initialize the model
    auto net = ChessNet();
    net->to(device);

    // 4) Load the trained weights
    std::string model_path = "../../training/NN_weights/model_V1.6_C_SMALL_FV_b_and_w_evals_scaled_10e_weighted_lr_1e4.pt";
    if (!load_model(net, model_path, device))
    {
        std::cerr << "Failed to load model. Exiting.\n";
        return 1;
    }

    // 5) Setup for testing
    const int test_batch_size = 1024;
    const int64_t training_dataset_size = 10'363'868;
    const int64_t validation_dataset_size = 1'295'484; 
    const int num_test_batches = validation_dataset_size / test_batch_size;
    int last_rowid = training_dataset_size + validation_dataset_size; // start of test data

    // Variables for accumulating metrics across batches
    double total_sse = 0.0;  // Sum of Squared Errors (for MSE)
    double total_mae = 0.0;  // Sum of Absolute Errors
    double sum_targets = 0.0;
    double sum_targets_sq = 0.0;
    int64_t total_samples = 0;

    // 6) Prepare CSV file for predictions/targets
    // Make sure to check you have enough space if your test set is very large.
    std::ofstream csv_file("predictions.csv");
    if (!csv_file.is_open())
    {
        std::cerr << "Failed to open predictions.csv for writing.\n";
        return 1;
    }
    // Write CSV header
    csv_file << "prediction,target\n";

    // 7) Loop over test batches
    for (int batch_idx = 0; batch_idx < num_test_batches; ++batch_idx)
    {
        BatchData batch_data = load_data(db, test_batch_size, last_rowid, net, device);
        last_rowid = batch_data.last_rowid;

        if (batch_data.inputs.size(0) == 0)
        {
            std::cerr << "No data found or invalid rows for batch " << batch_idx << std::endl;
            break;
        }

        torch::NoGradGuard no_grad;

        // Forward pass
        auto output = net->forward(batch_data.inputs); // shape [N, 1]
        auto predictions = output.squeeze(-1);         // shape [N]

        // Compute the raw errors
        auto errors = predictions - batch_data.targets;    // shape [N]

        // SSE (Sum of Squared Errors) for this batch
        double batch_sse = errors.pow(2).sum().item<double>();
        // Sum of absolute errors
        double batch_mae = errors.abs().sum().item<double>();

        // Update global sums
        total_sse += batch_sse;
        total_mae += batch_mae;

        // Accumulate target sums for R^2 calculation
        double batch_targets_sum = batch_data.targets.sum().item<double>();
        double batch_targets_sq_sum = batch_data.targets.pow(2).sum().item<double>();
        sum_targets += batch_targets_sum;
        sum_targets_sq += batch_targets_sq_sum;

        // Number of samples in this batch
        int64_t batch_size = batch_data.inputs.size(0);
        total_samples += batch_size;

        // Save predictions & targets to CSV
        // (Move to CPU if your tensors are on GPU)
        auto preds_cpu = predictions.to(torch::kCPU);
        auto targets_cpu = batch_data.targets.to(torch::kCPU);

        float* preds_data   = preds_cpu.data_ptr<float>();
        float* targets_data = targets_cpu.data_ptr<float>();

        for (int i = 0; i < batch_size; ++i)
        {
            csv_file << preds_data[i] << "," << targets_data[i] << "\n";
        }

        // (Optional) Print MSE for this batch
        double batch_mse = batch_sse / static_cast<double>(batch_size);
        std::cout << "Batch " << batch_idx
                  << " | MSE: " << batch_mse
                  << " | Targets Avg: " << batch_data.targets.mean().item<float>()
                  << " | Predictions Avg: " << predictions.mean().item<float>()
                  << std::endl;
    }

    // 8) Close the CSV file
    csv_file.close();

    // 9) Calculate final metrics over all tested samples
    if (total_samples > 0)
    {
        double avg_mse = total_sse / static_cast<double>(total_samples); 
        double rmse = std::sqrt(avg_mse);
        double mae = total_mae / static_cast<double>(total_samples);

        // For R^2 we need total sum of squares (TSS)
        double mean_target = sum_targets / static_cast<double>(total_samples);
        double total_tss = sum_targets_sq - (sum_targets * mean_target);

        // R^2 = 1 - SSE/TSS
        double r2 = 0.0;
        if (std::abs(total_tss) > 1e-15)
        {
            r2 = 1.0 - (total_sse / total_tss);
        }
        else
        {
            // Edge case: if all targets are identical
            r2 = (total_sse < 1e-15) ? 1.0 : 0.0;
        }

        std::cout << "========================================\n";
        std::cout << "Test Results over " << total_samples << " samples:\n";
        std::cout << "  * Average MSE: " << avg_mse << "\n";
        std::cout << "  * RMSE:        " << rmse   << "\n";
        std::cout << "  * MAE:         " << mae    << "\n";
        std::cout << "  * R^2 Score:   " << r2     << "\n";
        std::cout << "========================================\n";
    }
    else
    {
        std::cerr << "No samples were tested.\n";
    }

    // 10) Close DB
    sqlite3_close(db);

    return 0;
}




// #include <torch/torch.h>
// #include <sqlite3.h>
// #include <iostream>
// #include <cmath>
// #include "../../training/include/chessnet.h"    // Your ChessNet definition
// #include "../../training/include/data_loader.h" // Your load_data(...) function

// // This function loads a trained model from disk into 'net'
// bool load_model(ChessNet &net, const std::string &model_path, torch::Device device)
// {
//     try
//     {
//         torch::serialize::InputArchive input_archive;
//         input_archive.load_from(model_path);
//         net->load(input_archive);
//         net->to(device);
//         net->eval(); // Set model to eval mode for inference
//         std::cout << "Model weights loaded successfully from: " << model_path << std::endl;
//         return true;
//     }
//     catch (const c10::Error &e)
//     {
//         std::cerr << "Error loading model weights: " << e.what() << std::endl;
//         return false;
//     }
// }

// int main()
// {
//     // 1) Choose device
//     torch::Device device(torch::kCPU);
//     if (torch::cuda::is_available())
//     {
//         device = torch::kCUDA;
//         std::cout << "CUDA is available! Using GPU." << std::endl;
//     }
//     else
//     {
//         std::cout << "CUDA is not available. Using CPU." << std::endl;
//     }

//     // 2) Open the database
//     sqlite3 *db;
//     // Adjust path to your DB file if needed
//     if (sqlite3_open("../../data/chess_evals.db", &db))
//     {
//         std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
//         return 1;
//     }

//     // 3) Initialize the model
//     auto net = ChessNet();
//     net->to(device);

//     // 4) Load the trained weights
//     std::string model_path = "../../training/NN_weights/model_V1.5_C_FV_vlack_andwhite_evals_scaled_10e_weighted_lr_1e5.pt";
//     if (!load_model(net, model_path, device))
//     {
//         std::cerr << "Failed to load model. Exiting.\n";
//         return 1;
//     }

//     // 5) Setup for testing
//     const int test_batch_size = 1024;
//     const int64_t training_dataset_size = 10'363'868;
//     const int64_t validation_dataset_size = 1'295'484; 
//     const int num_test_batches = validation_dataset_size / test_batch_size;
//     int last_rowid = training_dataset_size + validation_dataset_size; // because first 1'295'484 is for validation

//     // Variables for accumulating metrics across batches
//     double total_sse = 0.0;  // Sum of Squared Errors (for MSE)
//     double total_mae = 0.0;  // Sum of Absolute Errors
//     double sum_targets = 0.0;
//     double sum_targets_sq = 0.0;
//     int64_t total_samples = 0;

//     // 6) Loop over test batches
//     for (int batch_idx = 0; batch_idx < num_test_batches; ++batch_idx)
//     {
//         BatchData batch_data = load_data(db, test_batch_size, last_rowid, net, device);
//         last_rowid = batch_data.last_rowid;

//         if (batch_data.inputs.size(0) == 0)
//         {
//             std::cerr << "No data found or invalid rows for batch " << batch_idx << std::endl;
//             break;
//         }

//         torch::NoGradGuard no_grad;

//         // Forward pass
//         auto output = net->forward(batch_data.inputs); // shape [N, 1]
//         auto predictions = output.squeeze(-1);         // shape [N]

//         // Compute the raw errors
//         auto errors = predictions - batch_data.targets;    // shape [N]

//         // SSE (Sum of Squared Errors) for this batch
//         double batch_sse = errors.pow(2).sum().item<double>();
//         // MAE (Mean Absolute Error) but first we accumulate sum of absolute errors
//         double batch_mae = errors.abs().sum().item<double>();

//         // Update global sums
//         total_sse += batch_sse;
//         total_mae += batch_mae;

//         // Accumulate target sums for R^2 calculation
//         double batch_targets_sum = batch_data.targets.sum().item<double>();
//         double batch_targets_sq_sum = batch_data.targets.pow(2).sum().item<double>();
//         sum_targets += batch_targets_sum;
//         sum_targets_sq += batch_targets_sq_sum;

//         // Number of samples in this batch
//         int64_t batch_size = batch_data.inputs.size(0);
//         total_samples += batch_size;

//         // You can still print MSE for this batch if desired
//         double batch_mse = batch_sse / static_cast<double>(batch_size);
//         std::cout << "Batch " << batch_idx
//                   << " | MSE: " << batch_mse
//                   << " | Targets Avg: " << batch_data.targets.mean().item<float>()
//                   << " | Predictions Avg: " << predictions.mean().item<float>()
//                   << std::endl;
//     }

//     // 7) Calculate final metrics over all tested samples
//     if (total_samples > 0)
//     {
//         double avg_mse = total_sse / static_cast<double>(total_samples); 
//         double rmse = std::sqrt(avg_mse);
//         double mae = total_mae / static_cast<double>(total_samples);

//         // For R^2 we need total sum of squares (TSS)
//         // TSS = sum( (y - mean(y))^2 ) = sum(y^2) - (sum(y)^2 / N)
//         double mean_target = sum_targets / static_cast<double>(total_samples);
//         double total_tss = sum_targets_sq - (sum_targets * mean_target);

//         // R^2 = 1 - SSE/TSS
//         // Make sure TSS is not zero (in case all targets are the same)
//         double r2 = 0.0;
//         if (std::abs(total_tss) > 1e-15) {
//             r2 = 1.0 - (total_sse / total_tss);
//         } 
//         else
//         {
//             // Edge case: if all targets are identical, R^2 is not well-defined.
//             // Could set it to 1.0 if SSE is also zero, or 0.0 otherwise.
//             r2 = (total_sse < 1e-15) ? 1.0 : 0.0;
//         }

//         // Print out final results
//         std::cout << "========================================\n";
//         std::cout << "Test Results over " << total_samples << " samples:\n";
//         std::cout << "  * Average MSE: " << avg_mse << "\n";
//         std::cout << "  * RMSE:        " << rmse   << "\n";
//         std::cout << "  * MAE:         " << mae    << "\n";
//         std::cout << "  * R^2 Score:   " << r2     << "\n";
//         std::cout << "========================================\n";
//     }
//     else
//     {
//         std::cerr << "No samples were tested.\n";
//     }

//     // 8) Close DB
//     sqlite3_close(db);

//     return 0;
// }

