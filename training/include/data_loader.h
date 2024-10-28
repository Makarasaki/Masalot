#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <sqlite3.h>
#include <cstdint>

struct ChessData {
    std::vector<std::vector<std::vector<int>>> bitboards;  // 3D vector: 14 bitboards, each 8x8
    float evaluation;
};

// Function to load chess positions and evaluations from the SQLite database
std::vector<ChessData> load_data(sqlite3* db, int batch_size, int epoch);

std::vector<std::vector<int>> intToBitboard(uint64_t bitboard);

std::vector<std::vector<int>> intToBitboardWhites(uint64_t bitboard);

std::vector<std::vector<int>> intToBitboardBlacks(uint64_t bitboard);

#endif  // DATA_LOADER_H
