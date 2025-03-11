// #ifndef CHESSNET_H
// #define CHESSNET_H

// #include <torch/torch.h>
// #include <sqlite3.h>
// #include <vector>
// #include <cstdint>

// struct ChessNet : torch::nn::Module {
//     torch::nn::Conv2d conv1, conv2, conv3, conv4;

//     torch::nn::BatchNorm2d bn1, bn2, bn3, bn4;

//     torch::nn::BatchNorm1d fc1_bn, fc2_bn, fc3_bn, conv_flat_bn;

//     torch::nn::Linear fc1, fc2, fc3;

//     ChessNet();

//     torch::Tensor forward(torch::Tensor x);

//     void initialize_weights();

//     static torch::Tensor bitboards_to_tensor(const std::vector<std::vector<std::vector<int>>>& bitboards);
// };

// #pragma once
// // Define a new linear chess network class that inherits from torch::nn::Module.
// class ChessNetLinear : public torch::nn::Module {
// public:
//     // Constructor
//     ChessNetLinear();

//     // Forward pass
//     torch::Tensor forward(torch::Tensor x);

//     torch::Tensor toTensor(const std::vector<int>& input_vector);

// private:
//     // Linear layers
//     torch::nn::Linear fc1;
//     torch::nn::Linear fc2;
//     torch::nn::Linear fc3;

//     // BatchNorm layers
//     torch::nn::BatchNorm1d bn1;
//     torch::nn::BatchNorm1d bn2;
//     torch::nn::BatchNorm1d bn3;

//     // Weight initialization
//     void initialize_weights();
// };

// // This macro makes it easy to create std::shared_ptr<ChessNetLinear>
// TORCH_MODULE(ChessNetLinear);

// #endif // CHESSNET_H

#ifndef CHESSNET_H
#define CHESSNET_H

#include <torch/torch.h>
#include <sqlite3.h>
#include <vector>
#include <cstdint>

struct BatchData
{
    torch::Tensor inputs;  // Shape: [batch_size, 837]
    torch::Tensor targets; // Shape: [batch_size]
    int64_t last_rowid;
};

struct ChessPosition
{
    const uint64_t WPawn;
    const uint64_t WKnight;
    const uint64_t WBishop;
    const uint64_t WRook;
    const uint64_t WQueen;
    const uint64_t WKing;

    const uint64_t BPawn;
    const uint64_t BKnight;
    const uint64_t BBishop;
    const uint64_t BRook;
    const uint64_t BQueen;
    const uint64_t BKing;

    const uint64_t EnPassant;
    const bool WhiteMove;

    const bool MyCastleL;
    const bool MyCastleR;

    const bool EnemyCastleL;
    const bool EnemyCastleR;

    // Constructor initializes all fields
    ChessPosition(uint64_t wP, uint64_t wN, uint64_t wB, uint64_t wR, uint64_t wQ, uint64_t wK,
                  uint64_t bP, uint64_t bN, uint64_t bB, uint64_t bR, uint64_t bQ, uint64_t bK,
                  uint64_t enp, bool whiteMv,
                  bool myCastL, bool myCastR,
                  bool enemyCastL, bool enemyCastR)
        : WPawn(wP), WKnight(wN), WBishop(wB), WRook(wR), WQueen(wQ), WKing(wK)

          ,
          BPawn(bP), BKnight(bN), BBishop(bB), BRook(bR), BQueen(bQ), BKing(bK)

          ,
          EnPassant(enp), WhiteMove(whiteMv), MyCastleL(myCastL), MyCastleR(myCastR), EnemyCastleL(enemyCastL), EnemyCastleR(enemyCastR)
    {
    }
};

std::vector<std::vector<int>> intToBitboard(uint64_t bitboard, int value);

// std::vector<std::vector<int>> intToBitboardWhites(uint64_t bitboard);

// std::vector<std::vector<int>> intToBitboardBlacks(uint64_t bitboard);

std::vector<int> intToVector64White(uint64_t bitboard);

// ----------------------------------------------
// Convolution-based ChessNet (updated as a class)
// ----------------------------------------------
class ChessNetConvImpl : public torch::nn::Module
{
public:
    // Constructor
    ChessNetConvImpl();

    // Forward pass
    torch::Tensor forward(torch::Tensor x);

    // Initialization
    void initialize_weights();

    // Optional static helper for converting bitboards to a tensor
    torch::Tensor toTensor(const ChessPosition &position);

private:
    // Convolutional layers
    torch::nn::Conv2d conv1, conv2, conv3, conv4;

    // BatchNorm2d for each conv layer
    torch::nn::BatchNorm2d bn1, bn2, bn3, bn4;

    // Flatten BN (and fully connected) layers
    torch::nn::BatchNorm1d fc1_bn, fc2_bn, fc3_bn, conv_flat_bn;

    // Fully connected layers
    torch::nn::Linear fc1, fc2, fc3;
};

// This macro will create a typedef: using ChessNet = std::shared_ptr<ChessNetConv>;
TORCH_MODULE(ChessNetConv);

// ----------------------------------------------
// Linear ChessNet
// ----------------------------------------------
class ChessNetLinearImpl : public torch::nn::Module
{
public:
    // Constructor
    ChessNetLinearImpl();

    // Forward pass
    torch::Tensor forward(torch::Tensor x);

    // Helper for converting a single 837-element vector to a Tensor
    torch::Tensor toTensor(const ChessPosition &position);

    std::vector<int> loadBitboard(uint64_t bitboard, bool isEnemy);

private:
    // Linear layers
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;

    // BatchNorm layers
    torch::nn::BatchNorm1d bn1;
    torch::nn::BatchNorm1d bn2;
    torch::nn::BatchNorm1d bn3;

    // Weight initialization
    void initialize_weights();
};

// This macro will create a typedef: using ChessNetLinear = std::shared_ptr<ChessNetLinear>;
TORCH_MODULE(ChessNetLinear);

// ChessPosition createChessPosition(const Board &board,
//                                   const BoardStatus &status,
//                                   const uint64_t &epTarget);

uint64_t flipVertical(uint64_t board, bool isWhite);

uint64_t rotate180(uint64_t board, bool isWhite);

// board and status to chess position



// Example enum for piece types
enum PieceType {
    PAWN   = 0,
    KNIGHT = 1,
    BISHOP = 2,
    ROOK   = 3,
    QUEEN  = 4,
    KING   = 5,
    NONE_PIECE = 6
};

enum MoveType {
    Kingmove = 0,
    KingCastle = 1,
    Pawnmove = 2,
    Pawnatk = 3,
    PawnEnpassantTake = 4,
    Pawnpush = 5,
    Pawnpromote = 6,
    Knightmove = 7,
    Bishopmove = 8,
    Rookmove = 9,
    Queenmove = 10
};

// ------------------------------------------------------------
// Function to calculate the HalfKP feature index
uint32_t calculateHalfKPIndex(uint64_t kingSquare, uint64_t pieceSquare,
                              PieceType pieceType, bool isWhite);
// ------------------------------------------------------------

// ------------------------------------------------------------
// Define the NNUE HalfKP network architecture
// ------------------------------------------------------------
struct NNUEHalfKPImpl : torch::nn::Module {
    // Default constructor (with some hard-coded values)
    NNUEHalfKPImpl();

    // Forward pass
    torch::Tensor forward(torch::Tensor x);

    // Initialize weights
    void initialize_weights();

    // Convert ChessPosition to a tensor of HalfKP feature indices
    torch::Tensor toTensor(const ChessPosition &position);

private:
    // Layers
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    torch::nn::BatchNorm1d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};

    // Function to calculate the HalfKP input vector
    std::vector<int64_t> createHalfKPInputVector(const ChessPosition &position);
};

// Torch’s macro that defines a module holder class (NNUEHalfKP)
// so you can instantiate it as `NNUEHalfKP model(...)`.
TORCH_MODULE(NNUEHalfKP);

// ----------------------------------------------
// Convolution-based ChessNet (updated as a class)
// ----------------------------------------------
class ChessNetConv2Impl : public torch::nn::Module
{
public:
    // Constructor
    ChessNetConv2Impl();

    // Forward pass
    torch::Tensor forward(torch::Tensor x);

    // Initialization
    void initialize_weights();

    // Optional static helper for converting bitboards to a tensor
    torch::Tensor toTensor(const ChessPosition &position);

private:
    // Convolutional layers
    torch::nn::Conv2d conv1, conv2;

    // BatchNorm2d for each conv layer
    torch::nn::BatchNorm2d bn1, bn2;

    // Flatten BN (and fully connected) layers
    torch::nn::BatchNorm1d fc1_bn, fc2_bn, fc3_bn, conv_flat_bn;

    // Fully connected layers
    torch::nn::Linear fc1, fc2, fc3;
};

// This macro will create a typedef: using ChessNet = std::shared_ptr<ChessNetConv>;
TORCH_MODULE(ChessNetConv2);

// using ChessNet = ChessNetLinear;
using ChessNet = ChessNetConv;
// using ChessNet = ChessNetConv2;

#endif // CHESSNET_H
