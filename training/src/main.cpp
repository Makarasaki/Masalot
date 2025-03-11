#include <torch/torch.h>
#include <sqlite3.h>
#include <iostream>
#include <fstream> // For file output
#include <vector>  // For storing losses
#include <cmath>   // For std::sqrt
#include "../include/chessnet.h"
#include "../include/data_loader.h"

//-------------------------------------------------
// Utility to save the model to disk
//-------------------------------------------------
void save_model(ChessNet &net, const torch::Device &device, const std::string &path)
{
    torch::serialize::OutputArchive output_archive;
    net->to(torch::kCPU); // Move model to CPU for saving
    net->save(output_archive);
    output_archive.save_to(path);
    net->to(device); // Move back to device (CPU or GPU)
    std::cout << "Model saved to " << path << std::endl;
}

//-------------------------------------------------
// Small helper to compute various metrics for a
// single batch: MSE, MAE, RMSE, R²
//-------------------------------------------------
struct BatchMetrics
{
    float mse;
    float mae;
    float rmse;
    float r2;
};

BatchMetrics compute_batch_metrics(const torch::Tensor &preds, const torch::Tensor &targets)
{
    // Ensure preds and targets have same shape
    auto preds_squeezed = preds.squeeze();
    auto targets_squeezed = targets.squeeze();

    // MSE
    auto mse_t = torch::mse_loss(preds_squeezed, targets_squeezed);
    float mse = mse_t.item<float>();

    // MAE
    auto mae_t = torch::mean(torch::abs(preds_squeezed - targets_squeezed));
    float mae = mae_t.item<float>();

    // RMSE
    float rmse = std::sqrt(mse);

    // R²
    // R² = 1 - SS_res / SS_tot
    // SS_res = sum((pred - actual)^2)
    // SS_tot = sum((actual - mean(actual))^2)
    float mean_targets = targets_squeezed.mean().item<float>();
    auto ss_res = torch::sum(torch::pow(preds_squeezed - targets_squeezed, 2));
    auto ss_tot = torch::sum(torch::pow(targets_squeezed - mean_targets, 2));
    float r2 = 1.0f - (ss_res.item<float>() / ss_tot.item<float>() + 1e-12f);
    // add small epsilon to avoid div-by-zero

    return {mse, mae, rmse, r2};
}

//-------------------------------------------------
// For epoch-level metrics, we do an "aggregator" so
// we can compute MSE, MAE, and R² over the entire
// dataset (not just per-batch averages).
//-------------------------------------------------
struct Aggregator
{
    double sum_abs_diff = 0.0;   // For MAE
    double sum_sq_diff = 0.0;    // For MSE
    double sum_targets = 0.0;    // For R²
    double sum_targets_sq = 0.0; // For R²
    int64_t total_samples = 0;

    void add_batch(const torch::Tensor &preds, const torch::Tensor &targets)
    {
        auto preds_squeezed = preds.squeeze();
        auto targets_squeezed = targets.squeeze();
        auto diffs = preds_squeezed - targets_squeezed;

        // Accumulate absolute differences
        sum_abs_diff += torch::sum(torch::abs(diffs)).item<double>();

        // Accumulate squared differences
        sum_sq_diff += torch::sum(diffs * diffs).item<double>();

        // Accumulate target sums
        sum_targets += torch::sum(targets_squeezed).item<double>();
        sum_targets_sq += torch::sum(targets_squeezed * targets_squeezed).item<double>();

        // Count samples
        total_samples += targets_squeezed.size(0);
    }

    // Compute final metrics
    BatchMetrics compute_metrics() const
    {
        if (total_samples == 0)
        {
            // Avoid dividing by zero
            return {0.f, 0.f, 0.f, 0.f};
        }
        // MSE & MAE
        float mse = static_cast<float>(sum_sq_diff / total_samples);
        float mae = static_cast<float>(sum_abs_diff / total_samples);
        float rmse = std::sqrt(mse);

        // R²
        // SS_res = sum_sq_diff
        // SS_tot = sum((y - mean(y))^2)
        double mean_targets_d = sum_targets / static_cast<double>(total_samples);
        double ss_tot = sum_targets_sq - static_cast<double>(total_samples) * mean_targets_d * mean_targets_d;
        float r2 = 1.f - static_cast<float>(sum_sq_diff / (ss_tot + 1e-12)); // epsilon for safety

        return {mse, mae, rmse, r2};
    }
};

//-------------------------------------------------
// Evaluate on the ENTIRE validation set, returning
// aggregated metrics. We do a smaller aggregator
// loop here instead of returning just MSE.
//-------------------------------------------------
BatchMetrics evaluate_on_validation_set(
    ChessNet &net,
    sqlite3 *db,
    int64_t validation_dataset_size,
    int64_t batch_size,
    torch::Device device)
{
    net->eval(); // Switch to eval mode
    torch::NoGradGuard no_grad;

    Aggregator aggregator;
    int64_t val_num_batches = validation_dataset_size / batch_size;
    int64_t val_last_rowid = 10'363'868;

    for (int i = 0; i < val_num_batches; ++i)
    {
        // Load from your validation table (adjust 'true/false' as needed)
        BatchData batch_data = load_data(db, batch_size, val_last_rowid, net, device);
        val_last_rowid = batch_data.last_rowid;

        if (batch_data.inputs.size(0) == 0)
            break;

        auto output = net->forward(batch_data.inputs);
        aggregator.add_batch(output, batch_data.targets);
    }

    net->train(); // Switch back to training
    return aggregator.compute_metrics();
}

int main()
{
    // -----------------------------
    // Device setup
    // -----------------------------
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

    // -----------------------------
    // Open Database
    // -----------------------------
    sqlite3 *db;
    if (sqlite3_open("../../data/chess_evals.db", &db))
    {
        std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
        return 1;
    }

    // -----------------------------
    // Hyperparameters
    // -----------------------------
    const int64_t num_epochs = 10;
    const int64_t batch_size = 1024;

    int id = 0;

    const int64_t training_dataset_size = 10'363'868;  // Example
    const int64_t validation_dataset_size = 1'295'484; // Example
    // const int64_t training_dataset_size = 1'363'868;  // Example
    // const int64_t validation_dataset_size = 1'00'000; // Example

    const int64_t num_batches = training_dataset_size / batch_size;

    float min_loss = std::numeric_limits<float>::max();
    float best_val_loss = std::numeric_limits<float>::max();

    std::string model_path = "../NN_weights/best_model.pt";

    // -----------------------------
    // Create Model + Optimizer
    // -----------------------------
    auto net = ChessNet();
    net->to(device);

    torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(1e-4));
    // int step_size = 1;   // Change learning rate every 1 epoch
    // double gamma = std::pow(1e-6 / 1e-4, 1.0 / 10.0);  // Compute gamma to smoothly reduce LR over 10 epochs
    
    // torch::optim::StepLR scheduler(optimizer, step_size, gamma);
    // torch::optim::StepLR scheduler(optimizer, /* step_size */ 2, /* gamma */ std::sqrt(0.1));

    // -----------------------------
    // CSV Files for Metrics
    // -----------------------------
    // 1) Per-Epoch (training + validation) -> epoch_metrics.csv
    // 2) Per-Batch (training only) -> batch_metrics.csv
    std::ofstream epoch_csv("epoch_metrics.csv");
    epoch_csv << "epoch,"
              << "train_mse,train_mae,train_rmse,train_r2,"
              << "val_mse,val_mae,val_rmse,val_r2\n";

    std::ofstream batch_csv("batch_metrics.csv");
    batch_csv << "epoch,batch,id,mse,mae,rmse,r2\n";

    // ==============================
    // (Optional) Load existing weights
    // ==============================
    // torch::serialize::InputArchive input_archive;
    // try {
    //     input_archive.load_from("../../training/NN_weights/model_last_w_5e.pt");
    //     net->load(input_archive);
    //     net->to(device);
    //     std::cout << "Model weights loaded successfully!\n";
    // } catch (const c10::Error &e) {
    //     std::cerr << "Error loading model weights: " << e.what() << std::endl;
    // }

    // ==============================
    // Initial Validation Test
    // ==============================
    {
        auto initial_val_metrics = evaluate_on_validation_set(net, db, validation_dataset_size, batch_size, device);
        epoch_csv << 0 << ","
                  << initial_val_metrics.mse << ","
                  << initial_val_metrics.mae << ","
                  << initial_val_metrics.rmse << ","
                  << initial_val_metrics.r2 << ","
                  << initial_val_metrics.mse << ","
                  << initial_val_metrics.mae << ","
                  << initial_val_metrics.rmse << ","
                  << initial_val_metrics.r2 << "\n";

        std::cout << "[Before Training] "
                  << "Val MSE=" << initial_val_metrics.mse << ", "
                  << "Val MAE=" << initial_val_metrics.mae << ", "
                  << "Val RMSE=" << initial_val_metrics.rmse << ", "
                  << "Val R2=" << initial_val_metrics.r2
                  << std::endl;
    }

    // ==============================
    // Training Loop
    // ==============================
    for (int epoch = 0; epoch < num_epochs; ++epoch)
    {
        std::cout << "Epoch " << (epoch + 1) << " Learning Rate: " << optimizer.param_groups()[0].options().get_lr() << std::endl;

        // -----------------------------
        // Reset rowid for new epoch
        // and aggregator for epoch
        // -----------------------------
        int64_t last_rowid = 0;
        Aggregator train_aggregator; // For epoch-level metrics (MSE, MAE, R^2, etc.)

        // -----------------------------
        // Train Batches
        // -----------------------------
        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx)
        {
            id ++;
            auto start_time = std::chrono::high_resolution_clock::now();

            BatchData batch_data = load_data(db, batch_size, last_rowid, net, device);
            last_rowid = batch_data.last_rowid;

            if (batch_data.inputs.size(0) == 0)
            {
                std::cerr << "No more training rows found.\n";
                break;
            }

            std::cout << "Target avg: " << batch_data.targets.mean().item().toFloat() << std::endl;
            std::cout << "Target range: "
                      << batch_data.targets.min().item().toFloat()
                      << " to "
                      << batch_data.targets.max().item().toFloat()
                      << std::endl;

            // Forward pass
            optimizer.zero_grad();
            auto output = net->forward(batch_data.inputs);

            // Compute loss (MSE)
            auto loss = torch::mse_loss(output.squeeze(), batch_data.targets);

            // Backprop + update
            loss.backward();
            optimizer.step();

            float current_loss = loss.item<float>();
            if (current_loss < min_loss)
            {
                min_loss = current_loss;
                save_model(net, device, model_path);
                std::cout << "New minimum (training) MSE = " << current_loss << " (model saved)\n";
            }

            // -----------------------------
            // Per-Batch Metrics
            // -----------------------------
            auto metrics = compute_batch_metrics(output, batch_data.targets);

            // Write to batch CSV: epoch, batch, MSE, MAE, RMSE, R^2
            batch_csv << (epoch + 1) << ","
                      << (batch_idx + 1) << ","
                      << id << ","
                      << metrics.mse << ","
                      << metrics.mae << ","
                      << metrics.rmse << ","
                      << metrics.r2 << "\n";

            // -----------------------------
            // Aggregator for full epoch
            // -----------------------------
            train_aggregator.add_batch(output, batch_data.targets);

            // Timing + debug info
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> duration = end_time - start_time;

            std::cout << "Epoch [" << (epoch + 1)
                      << " / " << num_epochs << "]  Batch [" << (batch_idx + 1)
                      << "/" << num_batches << "]  "
                      << "MSE: " << current_loss << "  "
                      << "Time: " << duration.count() << "s\n";
        }

        // -----------------------------
        // End of Epoch: Compute Training Metrics
        // -----------------------------
        auto train_metrics = train_aggregator.compute_metrics();

        // -----------------------------
        // Validation
        // -----------------------------
        auto val_metrics = evaluate_on_validation_set(net, db, validation_dataset_size, batch_size, device);

        // Display
        std::cout << "[Epoch " << (epoch + 1) << "] "
                  << "Train MSE=" << train_metrics.mse
                  << ", MAE=" << train_metrics.mae
                  << ", RMSE=" << train_metrics.rmse
                  << ", R2=" << train_metrics.r2 << " || "
                  << "Val MSE=" << val_metrics.mse
                  << ", MAE=" << val_metrics.mae
                  << ", RMSE=" << val_metrics.rmse
                  << ", R2=" << val_metrics.r2
                  << std::endl;

        // -----------------------------
        // Write per-epoch metrics to CSV
        // -----------------------------
        epoch_csv << (epoch + 1) << ","
                  << train_metrics.mse << ","
                  << train_metrics.mae << ","
                  << train_metrics.rmse << ","
                  << train_metrics.r2 << ","
                  << val_metrics.mse << ","
                  << val_metrics.mae << ","
                  << val_metrics.rmse << ","
                  << val_metrics.r2 << "\n";

        // -----------------------------
        // Save "best" model on Val MSE
        // -----------------------------
        if (val_metrics.mse < best_val_loss)
        {
            best_val_loss = val_metrics.mse;
            save_model(net, device, "../NN_weights/best_val_model.pt");
            std::cout << "New best validation MSE=" << best_val_loss << " (model saved)\n";
        }

        // -----------------------------
        // Save checkpoint at end of epoch
        // -----------------------------
        std::string epoch_model_path = "../NN_weights/model_epoch_" + std::to_string(epoch) + ".pt";
        save_model(net, device, epoch_model_path);

        // scheduler.step();
    }

    // -----------------------------
    // Save final model
    // -----------------------------
    save_model(net, device, "../NN_weights/model_last.pt");

    // Close CSVs & DB
    epoch_csv.close();
    batch_csv.close();
    sqlite3_close(db);

    return 0;
}
