#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <sqlite3.h>
#include <cstdint>
#include "../include/chessnet.h"

struct ChessData {
    // std::vector<std::vector<std::vector<int>>> bitboards;  // 3D vector: 14 bitboards, each 8x8
    std::vector<int> bitboards; // Flattened bitboards: 13 * 64 = 832 elements
    float evaluation;
};

// Function to load chess positions and evaluations from the SQLite database
BatchData load_data(sqlite3* db, int batch_size, int batch, ChessNet net, torch::Device device);

std::vector<std::vector<int>> intToBitboard(uint64_t bitboard);

std::vector<std::vector<int>> intToBitboardWhites(uint64_t bitboard);

std::vector<std::vector<int>> intToBitboardBlacks(uint64_t bitboard);

std::vector<int> intToVector64White(uint64_t bitboard);
std::vector<int> intToVector64Black(uint64_t bitboard);
std::vector<int> intToVector64(uint64_t bitboard);

#endif  // DATA_LOADER_H
