#include <cstdint>
#include <random>
#include "../include/zorbist.h"
#include "../giga/Chess_Base.hpp"



std::uint64_t ZOBRIST_PIECE[12][64];
std::uint64_t ZOBRIST_SIDE;
std::uint64_t ZOBRIST_CASTLING[16];
std::uint64_t ZOBRIST_EN_PASSANT[8];


// Helper to pop the least significant set bit from a bitboard:
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


void initZobristKeys() {
    std::mt19937_64 rng(1234567ULL); // or a truly random seed

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
 * Compute a Zobrist hash for the given board + status, 
 * also taking the en-passant target square as a separate argument.
 *
 * @param b               The current Board (piece bitboards).
 * @param st              The BoardStatus (side to move, castling, EP-flag).
 * @param enPassantSquare The square index [0..63] if an en-passant is possible, 
 *                        or -1 (or another sentinel) if no EP is available.
 * @return                64-bit Zobrist hash key for this position.
 */
std::uint64_t computeZobristHash(const Board &b, const BoardStatus &st, const uint64_t enPassantSquare)
{
    std::uint64_t hash = 0ULL;

    // Lambda to XOR in squares occupied by a particular piece bitboard
    auto accumulateBitboard = [&](std::uint64_t bitboard, int pieceIndex) {
        while (bitboard) {
            int sq = popLSB(bitboard);           // get the index of the LSB
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
    //    Typically, we XOR if it's Black to move. 
    //    If st.WhiteMove == false => black's turn => XOR
    if (!st.WhiteMove) {
        hash ^= ZOBRIST_SIDE;
    }

    // 4) Castling rights
    //    Combine bits for WCastleL, WCastleR, BCastleL, BCastleR into [0..15]
    int castlingIndex = 0;
    if (st.WCastleL) castlingIndex |= (1 << 0);
    if (st.WCastleR) castlingIndex |= (1 << 1);
    if (st.BCastleL) castlingIndex |= (1 << 2);
    if (st.BCastleR) castlingIndex |= (1 << 3);

    hash ^= ZOBRIST_CASTLING[castlingIndex];

    // 5) En-passant
    //    If st.HasEPPawn == true, we also check enPassantSquare
    //    to be a valid index in [0..63]. 
    //    Typically, we only need the 'file' to XOR into the hash.
    if (st.HasEPPawn && enPassantSquare >= 0 && enPassantSquare < 64)
    {
        int file = enPassantSquare & 7;  // or (enPassantSquare % 8), etc.
        hash ^= ZOBRIST_EN_PASSANT[file];
    }

    return hash;
}
