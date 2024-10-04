#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <sqlite3.h>
#include <cstdint>

// Struct to hold chess data
// struct ChessData {
//     std::vector<uint64_t> bitboards;  // Flattened bitboard data
//     float evaluation;
// };

struct ChessData {
    std::vector<std::vector<std::vector<int>>> bitboards;  // 3D vector: 14 bitboards, each 8x8
    float evaluation;
};


// Normalize evaluation function
float normalize_evaluation(float evaluation);

// Function to load chess positions and evaluations from the SQLite database
std::vector<ChessData> load_data(sqlite3* db, int batch_size, int epoch);

#endif  // DATA_LOADER_H
