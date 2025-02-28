#ifndef ZOBRIST_HPP
#define ZOBRIST_HPP

#include <cstdint>
#include <random>
#include "../giga/Chess_Base.hpp"

// Define piece indices for Zobrist hashing
enum {
    Z_WP = 0, Z_WN, Z_WB, Z_WR, Z_WQ, Z_WK,
    Z_BP, Z_BN, Z_BB, Z_BR, Z_BQ, Z_BK
};

// Zobrist hash tables
inline std::uint64_t ZOBRIST_PIECE[12][64];
inline std::uint64_t ZOBRIST_SIDE;
inline std::uint64_t ZOBRIST_CASTLING[16];
inline std::uint64_t ZOBRIST_EN_PASSANT[8];

/**
 * @brief Helper function to pop the least significant set bit from a bitboard.
 */
static inline int popLSB(std::uint64_t &bb) {
    unsigned long idx = 0;
#ifdef _MSC_VER
    _BitScanForward64(&idx, bb);
#else
    idx = __builtin_ctzll(bb);
#endif
    bb &= (bb - 1);  // clear that bit
    return static_cast<int>(idx);
}

/**
 * @brief Initializes Zobrist hash keys with random values.
 */
inline void initZobristKeys() {
    std::mt19937_64 rng(1234567ULL); // Seed for deterministic results

    for(int piece = 0; piece < 12; piece++) {
        for(int sq = 0; sq < 64; sq++) {
            ZOBRIST_PIECE[piece][sq] = rng();
        }
    }

    ZOBRIST_SIDE = rng();

    for(int i = 0; i < 16; i++) {
        ZOBRIST_CASTLING[i] = rng();
    }

    for(int f = 0; f < 8; f++) {
        ZOBRIST_EN_PASSANT[f] = rng();
    }
}

/**
 * @brief Computes the Zobrist hash for the given board position.
 * 
 * @param b               The board state (bitboards).
 * @param st              The board status (turn, castling rights, EP flag).
 * @param enPassantSquare The en-passant square (0-63) if available, -1 otherwise.
 * @return                64-bit Zobrist hash key for the position.
 */
inline std::uint64_t computeZobristHash(const Board &b, const BoardStatus &st, const uint64_t enPassantSquare) {
    std::uint64_t hash = 0ULL;

    // Lambda to XOR in squares occupied by a particular piece bitboard
    auto accumulateBitboard = [&](std::uint64_t bitboard, int pieceIndex) {
        while (bitboard) {
            int sq = popLSB(bitboard); // Get the index of the LSB
            hash ^= ZOBRIST_PIECE[pieceIndex][sq];
        }
    };

    // 1) White pieces
    accumulateBitboard(b.WPawn,   Z_WP);
    accumulateBitboard(b.WKnight, Z_WN);
    accumulateBitboard(b.WBishop, Z_WB);
    accumulateBitboard(b.WRook,   Z_WR);
    accumulateBitboard(b.WQueen,  Z_WQ);
    accumulateBitboard(b.WKing,   Z_WK);

    // 2) Black pieces
    accumulateBitboard(b.BPawn,   Z_BP);
    accumulateBitboard(b.BKnight, Z_BN);
    accumulateBitboard(b.BBishop, Z_BB);
    accumulateBitboard(b.BRook,   Z_BR);
    accumulateBitboard(b.BQueen,  Z_BQ);
    accumulateBitboard(b.BKing,   Z_BK);

    // 3) Side to move
    if (!st.WhiteMove) { // If Black's turn, XOR ZOBRIST_SIDE
        hash ^= ZOBRIST_SIDE;
    }

    // 4) Castling rights
    int castlingIndex = 0;
    if (st.WCastleL) castlingIndex |= (1 << 0);
    if (st.WCastleR) castlingIndex |= (1 << 1);
    if (st.BCastleL) castlingIndex |= (1 << 2);
    if (st.BCastleR) castlingIndex |= (1 << 3);

    hash ^= ZOBRIST_CASTLING[castlingIndex];

    // 5) En-passant
    if (st.HasEPPawn && enPassantSquare < 64) {
        int file = enPassantSquare & 7; // Get file index (0-7)
        hash ^= ZOBRIST_EN_PASSANT[file];
    }

    return hash;
}

#endif // ZOBRIST_HPP
