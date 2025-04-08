#ifndef ZOBRIST_H
#define ZOBRIST_H

#include <cstdint>
#include "../giga/Chess_Base.hpp"

// Define piece indices for Zobrist hashing
enum {
    Z_WP = 0, Z_WN, Z_WB, Z_WR, Z_WQ, Z_WK,
    Z_BP, Z_BN, Z_BB, Z_BR, Z_BQ, Z_BK
};

extern std::uint64_t ZOBRIST_PIECE[12][64];
extern std::uint64_t ZOBRIST_SIDE;
extern std::uint64_t ZOBRIST_CASTLING[16];
extern std::uint64_t ZOBRIST_EN_PASSANT[8];

// Function declarations

/**
 * @brief Initializes Zobrist hash keys with random values.
 */
void initZobristKeys();

/**
 * @brief Computes the Zobrist hash for the given board position.
 * 
 * @param b               The board state (bitboards).
 * @param st              The board status (turn, castling rights, EP flag).
 * @param enPassantSquare The en-passant square (0-63) if available, -1 otherwise.
 * @return                64-bit Zobrist hash key for the position.
 */
std::uint64_t computeZobristHash(const Board &b, const BoardStatus &st, const uint64_t enPassantSquare);

#endif // ZOBRIST_H
