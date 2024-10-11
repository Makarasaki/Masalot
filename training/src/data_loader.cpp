#include "../include/data_loader.h"
#include <iostream>

// Function to convert a 64-bit integer into an 8x8 bitboard (2D vector)
std::vector<std::vector<int>> intToBitboard(uint64_t bitboard) {
    std::vector<std::vector<int>> board(8, std::vector<int>(8, 0));
    for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 8; ++col) {
            // Extract each bit and place it in the 8x8 matrix
            board[row][col] = (bitboard >> (row * 8 + col)) & 1;
        }
    }
    return board;
}

std::vector<ChessData> load_data(sqlite3* db, int batch_size, int epoch) {
    std::vector<ChessData> data_batch;
    sqlite3_stmt* stmt;

    // Calculate the starting offset based on the epoch and batch size
    int offset = (epoch - 1) * batch_size;  // Start at the next chunk of data for each epoch

    // SQL query to select 14 bitboards and the evaluation with LIMIT and OFFSET
    const char* sql = "SELECT w_P_bitboard, w_N_bitboard, w_B_bitboard, w_R_bitboard, w_Q_bitboard, w_K_bitboard, "
                      "b_p_bitboard, b_n_bitboard, b_b_bitboard, b_r_bitboard, b_q_bitboard, b_k_bitboard, "
                      "en_passant_bitboard, info, eval_scaled FROM evaluations LIMIT ? OFFSET ?";

    // Prepare SQL statement
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "Failed to prepare SQL statement." << std::endl;
        return data_batch;
    }

    // Bind batch size and offset parameters
    sqlite3_bind_int(stmt, 1, batch_size);
    sqlite3_bind_int(stmt, 2, offset);

    // Fetch data row by row
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        ChessData entry;
        entry.bitboards.resize(14);  // 14 bitboards for each position

        // Convert each bitboard (64-bit integer) into an 8x8 vector
        for (int i = 0; i < 14; ++i) {
            uint64_t bitboard = static_cast<uint64_t>(sqlite3_column_int64(stmt, i));
            entry.bitboards[i] = intToBitboard(bitboard);  // Convert to 8x8 bitboard
        }

        // Get the evaluation value (scaled between -1 and 1)
        entry.evaluation = static_cast<float>(sqlite3_column_double(stmt, 14));
        // std::cout << entry.evaluation << std::endl;
        // Add the entry to the batch
        data_batch.push_back(entry);
    }

    // Finalize SQL statement
    sqlite3_finalize(stmt);

    return data_batch;
}

