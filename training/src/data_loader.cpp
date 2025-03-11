#include "../include/data_loader.h"
#include "../include/chessnet.h"
#include <iostream>

BatchData load_data(sqlite3 *db, int batch_size, int lastRowid, ChessNet net, torch::Device device)
{
    BatchData batch_data; // Will hold final (inputs, targets) Tensors
    std::vector<torch::Tensor> inputs;
    std::vector<torch::Tensor> targets;

    sqlite3_stmt *stmt;


    int offset = lastRowid;

    // SQL to retrieve rows
    const char *sql = 
        "SELECT w_P_bitboard, w_N_bitboard, w_B_bitboard, w_R_bitboard, w_Q_bitboard, w_K_bitboard, "
        "       b_p_bitboard, b_n_bitboard, b_b_bitboard, b_r_bitboard, b_q_bitboard, b_k_bitboard, "
        "       en_passant_bitboard, castling_KW, castling_QW, castling_kb, castling_qb, WhitesTurn, "
        "       eval_scaled, FEN, rowid "
        "FROM merged_shuffled_dataset "
        "WHERE rowid > ? "
        "ORDER BY rowid "
        "LIMIT ?";

    // Prepare statement
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK)
    {
        std::cerr << "Failed to prepare SQL statement: " << sqlite3_errmsg(db) << std::endl;
        return batch_data; // Returns empty BatchData
    }

    // Bind batch size and offset
    sqlite3_bind_int(stmt, 1, offset);
    sqlite3_bind_int(stmt, 2, batch_size);

    int row_count = 0;
    int64_t lastRowIdCaptured = 0;

    // Fetch rows
    while (sqlite3_step(stmt) == SQLITE_ROW && row_count < batch_size)
    {
        int64_t lastRowId = sqlite3_column_int64(stmt, 20);

        lastRowIdCaptured = lastRowId;
        // Build ChessPosition from current row
        bool whiteMove = static_cast<bool>(sqlite3_column_int(stmt, 17));

        ChessPosition position = [&]()
        {
            if (whiteMove)
            {
                return ChessPosition(
                    // White pieces
                    static_cast<uint64_t>(sqlite3_column_int64(stmt, 0)), // WPawn
                    static_cast<uint64_t>(sqlite3_column_int64(stmt, 1)), // WKnight
                    static_cast<uint64_t>(sqlite3_column_int64(stmt, 2)), // WBishop
                    static_cast<uint64_t>(sqlite3_column_int64(stmt, 3)), // WRook
                    static_cast<uint64_t>(sqlite3_column_int64(stmt, 4)), // WQueen
                    static_cast<uint64_t>(sqlite3_column_int64(stmt, 5)), // WKing

                    // Black pieces
                    static_cast<uint64_t>(sqlite3_column_int64(stmt, 6)),  // BPawn
                    static_cast<uint64_t>(sqlite3_column_int64(stmt, 7)),  // BKnight
                    static_cast<uint64_t>(sqlite3_column_int64(stmt, 8)),  // BBishop
                    static_cast<uint64_t>(sqlite3_column_int64(stmt, 9)),  // BRook
                    static_cast<uint64_t>(sqlite3_column_int64(stmt, 10)), // BQueen
                    static_cast<uint64_t>(sqlite3_column_int64(stmt, 11)), // BKing

                    // En passant bitboard
                    static_cast<uint64_t>(sqlite3_column_int64(stmt, 12)),

                    // WhiteMove
                    whiteMove,

                    // MyCastleL / MyCastleR
                    static_cast<bool>(sqlite3_column_int(stmt, 13)), // WCastleL
                    static_cast<bool>(sqlite3_column_int(stmt, 14)), // WCastleR

                    // EnemyCastleL / EnemyCastleR
                    static_cast<bool>(sqlite3_column_int(stmt, 15)), // BCastleL
                    static_cast<bool>(sqlite3_column_int(stmt, 16))  // BCastleR
                );
            }
            else
            {
                return ChessPosition(
                    // Black pieces first, but flipped
                    flipVertical(static_cast<uint64_t>(sqlite3_column_int64(stmt, 6)), whiteMove),  // BPawn
                    flipVertical(static_cast<uint64_t>(sqlite3_column_int64(stmt, 7)), whiteMove),  // BKnight
                    flipVertical(static_cast<uint64_t>(sqlite3_column_int64(stmt, 8)), whiteMove),  // BBishop
                    flipVertical(static_cast<uint64_t>(sqlite3_column_int64(stmt, 9)), whiteMove),  // BRook
                    flipVertical(static_cast<uint64_t>(sqlite3_column_int64(stmt, 10)), whiteMove), // BQueen
                    flipVertical(static_cast<uint64_t>(sqlite3_column_int64(stmt, 11)), whiteMove), // BKing

                    // Then white pieces, but flipped
                    flipVertical(static_cast<uint64_t>(sqlite3_column_int64(stmt, 0)), whiteMove), // WPawn
                    flipVertical(static_cast<uint64_t>(sqlite3_column_int64(stmt, 1)), whiteMove), // WKnight
                    flipVertical(static_cast<uint64_t>(sqlite3_column_int64(stmt, 2)), whiteMove), // WBishop
                    flipVertical(static_cast<uint64_t>(sqlite3_column_int64(stmt, 3)), whiteMove), // WRook
                    flipVertical(static_cast<uint64_t>(sqlite3_column_int64(stmt, 4)), whiteMove), // WQueen
                    flipVertical(static_cast<uint64_t>(sqlite3_column_int64(stmt, 5)), whiteMove), // WKing

                    // En passant bitboard
                    flipVertical(static_cast<uint64_t>(sqlite3_column_int64(stmt, 12)), whiteMove),

                    // WhiteMove
                    whiteMove,

                    // Now BCastleL/BCastleR become MyCastleL/MyCastleR
                    static_cast<bool>(sqlite3_column_int(stmt, 15)), // BCastleL
                    static_cast<bool>(sqlite3_column_int(stmt, 16)), // BCastleR

                    // And WCastleL/WCastleR become EnemyCastleL/EnemyCastleR
                    static_cast<bool>(sqlite3_column_int(stmt, 13)), // WCastleL
                    static_cast<bool>(sqlite3_column_int(stmt, 14))  // WCastleR
                );
            }
        }();

        // Load evaluation (column 18)
        float evaluation = static_cast<float>(sqlite3_column_double(stmt, 18));
        // positive means good position for side that is now doing move
        // so flip the evaluatiion around 0 if it's blacks move
        evaluation *= whiteMove ? 1.0f : -1.0f;

        torch::Tensor input_tensor;
        try
        {
            input_tensor = net->toTensor(position);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error converting position to tensor: " << e.what() << std::endl;
            continue; // Skip invalid row
        }

        input_tensor = input_tensor.unsqueeze(0);

        // Move input_tensor to device (CPU or CUDA)
        input_tensor = input_tensor.to(device);

        // Collect all input samples
        inputs.push_back(input_tensor);

        torch::Tensor target_tensor = torch::tensor(evaluation, torch::dtype(torch::kFloat32)).to(device);
        targets.push_back(target_tensor);

        row_count++;
    }

    sqlite3_finalize(stmt);

    // If no rows were fetched, return empty BatchData
    if (row_count == 0)
    {
        std::cerr << "No data found in database or no valid rows." << std::endl;
        return batch_data;
    }

    batch_data.inputs = torch::cat(inputs, /*dim=*/0);

    batch_data.targets = torch::stack(targets, /*dim=*/0).squeeze(-1);

    std::cout << "last rowid:" <<  lastRowIdCaptured << std::endl;

    batch_data.last_rowid = lastRowIdCaptured;

    return batch_data;
}