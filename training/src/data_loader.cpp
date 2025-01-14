#include "../include/data_loader.h"
#include "../include/chessnet.h"
#include <iostream>

// Function to convert a 64-bit integer into an 8x8 bitboard (2D vector)
std::vector<std::vector<int>> intToBitboard(uint64_t bitboard)
{
    std::vector<std::vector<int>> board(8, std::vector<int>(8, 0));
    for (int row = 0; row < 8; row++)
    {
        for (int col = 0; col < 8; col++)
        {
            // Extract each bit and place it in the 8x8 matrix
            board[row][col] = (bitboard >> (row * 8 + col)) & 1;
        }
    }
    return board;
}

std::vector<std::vector<int>> intToBitboardWhites(uint64_t bitboard)
{
    std::vector<std::vector<int>> board(8, std::vector<int>(8, 0));
    // std::cout << "INT to w bitboard: " << std::endl;
    for (int row = 0; row < 8; row++)
    {
        for (int col = 0; col < 8; col++)
        {
            // Extract each bit and place it in the 8x8 matrix
            board[row][col] = (bitboard >> (row * 8 + col)) & 1;
            // std::cout << board[row][col];
        }
        // std::cout << std::endl;
    }
    return board;
}

std::vector<std::vector<int>> intToBitboardBlacks(uint64_t bitboard)
{
    std::vector<std::vector<int>> board(8, std::vector<int>(8, 0));
    for (int row = 0; row < 8; row++)
    {
        for (int col = 0; col < 8; col++)
        {
            // Extract each bit and place it in the 8x8 matrix
            board[row][col] = (bitboard >> (row * 8 + col)) & 1 ? -1 : 0;
        }
    }
    return board;
}

// White bitboard -> +1 for each set bit
std::vector<int> intToVector64White(uint64_t bitboard)
{
    std::vector<int> vec(64, 0);
    for (int i = 0; i < 64; i++)
    {
        // Check if bit i is set, then store +1 in that position
        vec[i] = ((bitboard >> i) & 1ULL) ? 1 : 0;
    }
    return vec;
}

// Black bitboard -> -1 for each set bit
std::vector<int> intToVector64Black(uint64_t bitboard)
{
    std::vector<int> vec(64, 0);
    for (int i = 0; i < 64; i++)
    {
        // Check if bit i is set, then store -1 in that position
        vec[i] = ((bitboard >> i) & 1ULL) ? -1 : 0;
    }
    return vec;
}

// Neutral (e.g., en-passant) -> +1 for each set bit
std::vector<int> intToVector64(uint64_t bitboard)
{
    std::vector<int> vec(64, 0);
    for (int i = 0; i < 64; i++)
    {
        // Check if bit i is set, then store +1 in that position
        vec[i] = ((bitboard >> i) & 1ULL) ? 1 : 0;
    }
    return vec;
}

std::vector<std::vector<std::vector<int>>> info_to_bitboards(int info)
{
    // Create a vector to hold the 8x8 bitboards for each bit
    std::vector<std::vector<std::vector<int>>> bitboards;

    // Loop through each bit and create an 8x8 matrix based on the bit value
    for (int bit = 0; bit < 5; bit++)
    {
        bool bit_value = (info >> bit) & 1;
        // std::cout << "info bits" << bit_value << std::endl;
        // Create an 8x8 matrix filled with the bit value
        std::vector<std::vector<int>> bitboard(8, std::vector<int>(8, bit_value ? 1 : 0));
        bitboards.push_back(bitboard);
    }

    return bitboards;
}

BatchData load_data(sqlite3 *db, int batch_size, int batch, ChessNet net, torch::Device device)
{
    BatchData batch_data; // Will hold final (inputs, targets) Tensors
    std::vector<torch::Tensor> inputs;
    std::vector<torch::Tensor> targets;

    sqlite3_stmt *stmt;

    // Calculate offset for pagination
    int offset = batch * batch_size;

    // SQL to retrieve rows
    const char *sql =
        "SELECT w_P_bitboard, w_N_bitboard, w_B_bitboard, w_R_bitboard, w_Q_bitboard, w_K_bitboard, "
        "       b_p_bitboard, b_n_bitboard, b_b_bitboard, b_r_bitboard, b_q_bitboard, b_k_bitboard, "
        "       en_passant_bitboard, castling_KW, castling_QW, castling_kb, castling_qb, WhitesTurn, "
        "       eval_scaled, FEN "
        "FROM training_dataset "
        "LIMIT ? OFFSET ?";

    // Prepare statement
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK)
    {
        std::cerr << "Failed to prepare SQL statement: " << sqlite3_errmsg(db) << std::endl;
        return batch_data; // Returns empty BatchData
    }

    // Bind batch size and offset
    sqlite3_bind_int(stmt, 1, batch_size);
    sqlite3_bind_int(stmt, 2, offset);

    int row_count = 0;

    // Fetch rows
    while (sqlite3_step(stmt) == SQLITE_ROW && row_count < batch_size)
    {
        // Build ChessPosition from current row
        bool whiteMove = static_cast<bool>(sqlite3_column_int(stmt, 17));

        ChessPosition position = {
            // White pieces
            rotate180(static_cast<uint64_t>(sqlite3_column_int64(stmt, 0)), whiteMove), // WPawn
            rotate180(static_cast<uint64_t>(sqlite3_column_int64(stmt, 1)), whiteMove), // WKnight
            rotate180(static_cast<uint64_t>(sqlite3_column_int64(stmt, 2)), whiteMove), // WBishop
            rotate180(static_cast<uint64_t>(sqlite3_column_int64(stmt, 3)), whiteMove), // WRook
            rotate180(static_cast<uint64_t>(sqlite3_column_int64(stmt, 4)), whiteMove), // WQueen
            rotate180(static_cast<uint64_t>(sqlite3_column_int64(stmt, 5)), whiteMove), // WKing

            // Black pieces
            rotate180(static_cast<uint64_t>(sqlite3_column_int64(stmt, 6)), whiteMove),  // BPawn
            rotate180(static_cast<uint64_t>(sqlite3_column_int64(stmt, 7)), whiteMove),  // BKnight
            rotate180(static_cast<uint64_t>(sqlite3_column_int64(stmt, 8)), whiteMove),  // BBishop
            rotate180(static_cast<uint64_t>(sqlite3_column_int64(stmt, 9)), whiteMove),  // BRook
            rotate180(static_cast<uint64_t>(sqlite3_column_int64(stmt, 10)), whiteMove), // BQueen
            rotate180(static_cast<uint64_t>(sqlite3_column_int64(stmt, 11)), whiteMove), // BKing

            // En passant bitboard
            rotate180(static_cast<uint64_t>(sqlite3_column_int64(stmt, 12)), whiteMove),

            // WhiteMove
            whiteMove,

            // WCastleL
            whiteMove
                ? static_cast<bool>(sqlite3_column_int(stmt, 13))  // WCastleL
                : static_cast<bool>(sqlite3_column_int(stmt, 15)), // BCastleL

            // WCastleR
            whiteMove
                ? static_cast<bool>(sqlite3_column_int(stmt, 14))  // WCastleR
                : static_cast<bool>(sqlite3_column_int(stmt, 16)), // BCastleR

            // BCastleL
            whiteMove
                ? static_cast<bool>(sqlite3_column_int(stmt, 15))  // BCastleL
                : static_cast<bool>(sqlite3_column_int(stmt, 13)), // WCastleL

            // BCastleR
            whiteMove
                ? static_cast<bool>(sqlite3_column_int(stmt, 16)) // BCastleR
                : static_cast<bool>(sqlite3_column_int(stmt, 14)) // WCastleR
        };
        // Load evaluation (column 18)
        float evaluation = static_cast<float>(sqlite3_column_double(stmt, 18));
        // positive means good position for side that is now doing move
        // so flip the evaluatiion around 0 if it's blacks move 
        evaluation *= whiteMove ? 1.0f : -1.0f;

        // Convert ChessPosition to a [837]-sized Tensor
        torch::Tensor input_tensor;
        try
        {
            // Now that 'net' is a shared pointer, call member functions via '->'
            input_tensor = net->toTensor(chess_position); // shape [837]
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error converting position to tensor: " << e.what() << std::endl;
            continue; // Skip invalid row
        }

        // Add batch dimension: shape => [1, 837]
        input_tensor = input_tensor.unsqueeze(0);

        // Move input_tensor to device (CPU or CUDA)
        input_tensor = input_tensor.to(device);

        // Collect all input samples
        inputs.push_back(input_tensor);

        // Make a 1D target tensor for the evaluation
        torch::Tensor target_tensor = torch::tensor(evaluation, torch::dtype(torch::kFloat32)).to(device);
        targets.push_back(target_tensor);

        row_count++;
    }

    // Finalize the statement
    sqlite3_finalize(stmt);

    // If no rows were fetched, return empty BatchData
    if (row_count == 0)
    {
        std::cerr << "No data found in database or no valid rows." << std::endl;
        return batch_data;
    }

    // Concatenate input Tensors into shape [row_count, 837]
    batch_data.inputs = torch::cat(inputs, /*dim=*/0);

    // Stack target Tensors into shape [row_count]
    batch_data.targets = torch::stack(targets, /*dim=*/0).squeeze(-1);

    // Debug: show first row or shapes if needed
    // std::cout << batch_data.inputs[0] << std::endl;

    return batch_data;
}

// std::vector<ChessData> load_data(sqlite3* db, int batch_size, int batch) {
//     std::vector<ChessData> data_batch;
//     sqlite3_stmt* stmt;

//     // Calculate the starting offset based on the epoch and batch size
//     int offset = batch * batch_size;  // Start at the next chunk of data for each epoch

//     // REMEMBER ABOUT SPACES
//     const char* sql = "SELECT w_P_bitboard, w_N_bitboard, w_B_bitboard, w_R_bitboard, w_Q_bitboard, w_K_bitboard, "
//                   "b_p_bitboard, b_n_bitboard, b_b_bitboard, b_r_bitboard, b_q_bitboard, b_k_bitboard, "
//                   "en_passant_bitboard, castling_KW, castling_QW, castling_kb, castling_qb, WhitesTurn, eval_scaled, FEN "
//                   "FROM evaluations_rand "
//                   "WHERE WhitesTurn = 1 "
//                   "LIMIT ? OFFSET ?";

//     // Prepare SQL statement
//     if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
//         std::cerr << "Failed to prepare SQL statement." << std::endl;
//         return data_batch;
//     }

//     // Bind batch size and offset parameters
//     sqlite3_bind_int(stmt, 1, batch_size);
//     sqlite3_bind_int(stmt, 2, offset);

//     // Fetch data row by row
//     while (sqlite3_step(stmt) == SQLITE_ROW) {
//         ChessData entry;
//         uint64_t allOnes = ~0ULL;
//         entry.bitboards.resize(13);  // 14 bitboards for each position

//         // Convert each bitboard (64-bit integer) into an 8x8 vector
//         for (int i = 0; i < 6; ++i) {
//             uint64_t bitboard = static_cast<uint64_t>(sqlite3_column_int64(stmt, i));
//             entry.bitboards[i] = intToBitboardWhites(bitboard);  // Convert to 8x8 bitboard
//         }

//         for (int i = 6; i < 12; ++i) {
//             uint64_t bitboard = static_cast<uint64_t>(sqlite3_column_int64(stmt, i));
//             entry.bitboards[i] = intToBitboardBlacks(bitboard);  // Convert to 8x8 bitboard
//         }

//         entry.bitboards[12] = intToBitboard(static_cast<uint64_t>(sqlite3_column_int64(stmt, 12)));  // En Passant

//         entry.evaluation = static_cast<float>(sqlite3_column_double(stmt, 18));
//         // std::cout << entry.evaluation << std::endl;
//         // Add the entry to the batch
//         data_batch.push_back(entry);
//     }

//     // Finalize SQL statement
//     sqlite3_finalize(stmt);

//     return data_batch;
// }
