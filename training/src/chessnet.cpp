#include "../include/chessnet.h"
#include <iostream>

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

// ------------------------------------------
// ChessNetConv (Convolution-based)
// ------------------------------------------
ChessNetConvImpl::ChessNetConvImpl()
    : conv1(torch::nn::Conv2dOptions(13, 64, 3).stride(1).padding(1)),
      conv2(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
      conv3(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
      conv4(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1)),

      bn1(64), bn2(128), bn3(256), bn4(512),

      conv_flat_bn(512 * 8 * 8),

      fc1(512 * 8 * 8, 1024),
      fc1_bn(1024),
      fc2(1024, 256),
      fc2_bn(256),
      fc3_bn(1),
      fc3(256, 1)
{
    // Register modules
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("conv4", conv4);

    register_module("bn1", bn1);
    register_module("bn2", bn2);
    register_module("bn3", bn3);
    register_module("bn4", bn4);

    register_module("conv_flat_bn", conv_flat_bn);

    register_module("fc1", fc1);
    register_module("fc1_bn", fc1_bn);
    register_module("fc2", fc2);
    register_module("fc2_bn", fc2_bn);
    register_module("fc3_bn", fc3_bn);
    register_module("fc3", fc3);

    // Initialize weights
    initialize_weights();
}

torch::Tensor ChessNetConvImpl::forward(torch::Tensor x)
{
    // Convolution + BN + ReLU
    x = torch::relu(bn1(conv1(x)));
    x = torch::relu(bn2(conv2(x)));
    x = torch::relu(bn3(conv3(x)));
    x = torch::relu(bn4(conv4(x)));

    // Flatten from [batch_size, 512, 8, 8] to [batch_size, 512*8*8]
    x = x.view({-1, 512 * 8 * 8});

    // BatchNorm on flattened conv output
    x = conv_flat_bn(x);

    // FC1 -> BN -> ReLU
    x = torch::relu(fc1_bn(fc1(x)));

    // FC2 -> BN -> ReLU
    x = torch::relu(fc2_bn(fc2(x)));

    // FC3 -> BN -> Tanh => output in [-1, 1]
    x = torch::tanh(fc3_bn(fc3(x)));

    // Debug: if batch is large, print stats
    if (x.size(0) > 100)
    {
        std::cout << "output avg: " << x.mean().item().toFloat() << std::endl;
        std::cout << "output range: " << x.min().item().toFloat()
                  << " to " << x.max().item().toFloat() << std::endl;
    }

    return x;
}

void ChessNetConvImpl::initialize_weights()
{
    for (auto &module : modules(/*include_self=*/false))
    {
        if (auto *conv = dynamic_cast<torch::nn::Conv2dImpl *>(module.get()))
        {
            torch::nn::init::kaiming_uniform_(
                conv->weight, /*a=*/0.01,
                torch::kFanIn,
                torch::kLeakyReLU);
            if (conv->options.bias())
            {
                torch::nn::init::constant_(conv->bias, 0.01);
            }
        }
        else if (auto *fc = dynamic_cast<torch::nn::LinearImpl *>(module.get()))
        {
            torch::nn::init::kaiming_uniform_(
                fc->weight, /*a=*/0.01,
                torch::kFanIn,
                torch::kLeakyReLU);
            if (fc->options.bias())
            {
                torch::nn::init::constant_(fc->bias, 0.01);
            }
        }
        else if (auto *bn1d = dynamic_cast<torch::nn::BatchNorm1dImpl *>(module.get()))
        {
            torch::nn::init::ones_(bn1d->weight);
            torch::nn::init::zeros_(bn1d->bias);
        }
        else if (auto *bn2d = dynamic_cast<torch::nn::BatchNorm2dImpl *>(module.get()))
        {
            torch::nn::init::ones_(bn2d->weight);
            torch::nn::init::zeros_(bn2d->bias);
        }
    }
}

torch::Tensor ChessNetConvImpl::toTensor(const ChessPosition &position)
{

    // Convert each 64-bit bitboard into an 8x8 2D vector of ints
    auto wPawn    = intToBitboardWhites(position.WPawn);
    auto wKnight  = intToBitboardWhites(position.WKnight);
    auto wBishop  = intToBitboardWhites(position.WBishop);
    auto wRook    = intToBitboardWhites(position.WRook);
    auto wQueen   = intToBitboardWhites(position.WQueen);
    auto wKing    = intToBitboardWhites(position.WKing);

    auto bPawn    = intToBitboardBlacks(position.BPawn);
    auto bKnight  = intToBitboardBlacks(position.BKnight);
    auto bBishop  = intToBitboardBlacks(position.BBishop);
    auto bRook    = intToBitboardBlacks(position.BRook);
    auto bQueen   = intToBitboardBlacks(position.BQueen);
    auto bKing    = intToBitboardBlacks(position.BKing);

    auto enPassant = intToBitboard(position.EnPassant); // neutral => 0 or 1

    // We collect them into a single container for easier iteration
    // Each element is an 8x8 matrix (std::vector<std::vector<int>>)
    std::vector<std::vector<std::vector<int>>> allBoards = {
        wPawn, wKnight, wBishop, wRook, wQueen, wKing,
        bPawn, bKnight, bBishop, bRook, bQueen, bKing,
        enPassant
    };

    // Create a float32 tensor of shape [13, 8, 8]
    auto channels = static_cast<int>(allBoards.size()); // should be 13
    torch::Tensor tensor = torch::empty({channels, 8, 8}, torch::kFloat32);
    auto accessor = tensor.accessor<float, 3>();

    // Copy data from each 8×8 board into the tensor
    for (int c = 0; c < channels; ++c)
    {
        for (int row = 0; row < 8; ++row)
        {
            for (int col = 0; col < 8; ++col)
            {
                accessor[c][row][col] = static_cast<float>(allBoards[c][row][col]);
            }
        }
    }
    return tensor;
}

// ------------------------------------------
// ChessNetLinear (Linear-based)
// ------------------------------------------
ChessNetLinearImpl::ChessNetLinearImpl()
    : fc1(837, 512),
      fc2(512, 256),
      fc3(256, 1),
      bn1(512),
      bn2(256),
      bn3(1)
{
    // Register modules
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);

    register_module("bn1", bn1);
    register_module("bn2", bn2);
    register_module("bn3", bn3);

    initialize_weights();
}

torch::Tensor ChessNetLinearImpl::forward(torch::Tensor x)
{
    // If shape is [837], unsqueeze to [1, 837]
    if (x.dim() == 1)
    {
        x = x.unsqueeze(0);
    }

    // fc1 -> bn1 -> relu
    x = fc1(x);
    x = bn1(x);
    x = torch::relu(x);

    // fc2 -> bn2 -> relu
    x = fc2(x);
    x = bn2(x);
    x = torch::relu(x);

    // fc3 -> bn3 -> tanh
    x = fc3(x);
    x = bn3(x);
    x = torch::tanh(x);

    // Debug: print stats
    std::cout << "output avg: " << x.mean().item().toFloat() << std::endl;
    std::cout << "output range: " << x.min().item().toFloat()
              << " to " << x.max().item().toFloat() << std::endl;

    // shape: [batch_size, 1]
    return x;
}

void ChessNetLinearImpl::initialize_weights()
{
    for (auto &module : modules(/*include_self=*/false))
    {
        // For each Linear layer
        if (auto *fc = dynamic_cast<torch::nn::LinearImpl *>(module.get()))
        {
            torch::nn::init::kaiming_uniform_(
                fc->weight,
                /*a=*/0.01,
                torch::kFanIn,
                torch::kLeakyReLU);
            if (fc->options.bias())
            {
                torch::nn::init::constant_(fc->bias, 0.01);
            }
        }
        // For BatchNorm1d layers
        else if (auto *bn1d = dynamic_cast<torch::nn::BatchNorm1dImpl *>(module.get()))
        {
            torch::nn::init::ones_(bn1d->weight);
            torch::nn::init::zeros_(bn1d->bias);
        }
    }
}

// torch::Tensor ChessNetLinear::toTensor(const std::vector<int> &input_vector)
// {
//     // Must have 837 elements
//     if (input_vector.size() != 837)
//     {
//         throw std::runtime_error(
//             "ChessNetLinear::toTensor: input vector must have exactly 837 elements!");
//     }

//     torch::Tensor tensor = torch::empty({837}, torch::kFloat32);
//     auto accessor = tensor.accessor<float, 1>();

//     for (int i = 0; i < 837; i++)
//     {
//         accessor[i] = static_cast<float>(input_vector[i]);
//     }

//     // shape: [837]
//     return tensor;
// }

std::vector<int> ChessNetLinearImpl::loadBitboard(uint64_t bitboard, bool isEnemy)
{
    // "my" pieces set as 1, enemy pieces set as -1
    // empty squares are 0
    int value = isEnemy ? -1 : 1; // Use ternary operator for brevity

    std::vector<int> vec(64, 0);
    for (int i = 0; i < 64; i++)
    {
        vec[i] = ((bitboard >> i) & 1ULL) ? value : 0; // Set value if the bit is 1
    }

    return vec;
}

torch::Tensor ChessNetLinearImpl::toTensor(const ChessPosition &position)
{
    std::vector<int> bitboards(837, 0);
    int offset = 0;

    // Helper lambda to process each bitboard field
    auto process = [&](uint64_t bb, bool isEnemy)
    {
        std::vector<int> flat = loadBitboard(bb, isEnemy);
        if (flat.size() != 64)
        {
            throw std::runtime_error("Error: Flattened bitboard size is not 64.");
        }
        std::copy(flat.begin(), flat.end(), bitboards.begin() + offset);
        offset += 64;
    };

    // Process white pieces (assuming isEnemy = !WhiteMove for white pieces)
    process(position.WPawn, !position.WhiteMove);
    process(position.WKnight, !position.WhiteMove);
    process(position.WBishop, !position.WhiteMove);
    process(position.WRook, !position.WhiteMove);
    process(position.WQueen, !position.WhiteMove);
    process(position.WKing, !position.WhiteMove);

    // Process black pieces (assuming isEnemy = WhiteMove for black pieces)
    process(position.BPawn, position.WhiteMove);
    process(position.BKnight, position.WhiteMove);
    process(position.BBishop, position.WhiteMove);
    process(position.BRook, position.WhiteMove);
    process(position.BQueen, position.WhiteMove);
    process(position.BKing, position.WhiteMove);

    // Process en passant; assume en passant squares are neutral (not enemy)
    process(position.EnPassant, false);

    // Add castling rights and WhiteMove as final five entries
    bitboards[832] = position.MyCastleL ? 1 : 0;
    bitboards[833] = position.MyCastleR ? 1 : 0;
    bitboards[834] = position.EnemyCastleL ? 1 : 0;
    bitboards[835] = position.EnemyCastleR ? 1 : 0;
    bitboards[836] = position.WhiteMove ? 1 : 0;

    // Must have 837 elements
    if (bitboards.size() != 837)
    {
        throw std::runtime_error("ChessNetLinear::toTensor: input vector must have exactly 837 elements!");
    }

    torch::Tensor tensor = torch::empty({837}, torch::kFloat32);
    auto accessor = tensor.accessor<float, 1>();

    for (int i = 0; i < 837; i++)
    {
        accessor[i] = static_cast<float>(bitboards[i]);
    }

    // shape: [837]
    if (torch::cuda::is_available())
    {
        tensor = tensor.to(torch::kCUDA);
    }
    return tensor;
}

uint64_t rotate180(uint64_t board, bool isWhite)
{
    if (isWhite)
    {
        return board; // Do nothing for white pieces
    }

    const uint64_t h1 = 0x5555555555555555ULL;
    const uint64_t h2 = 0x3333333333333333ULL;
    const uint64_t h4 = 0x0F0F0F0F0F0F0F0FULL;
    const uint64_t v1 = 0x00FF00FF00FF00FFULL;
    const uint64_t v2 = 0x0000FFFF0000FFFFULL;

    board = ((board >> 1) & h1) | ((board & h1) << 1);   // Swap adjacent bits
    board = ((board >> 2) & h2) | ((board & h2) << 2);   // Swap pairs of bits
    board = ((board >> 4) & h4) | ((board & h4) << 4);   // Swap nibbles
    board = ((board >> 8) & v1) | ((board & v1) << 8);   // Swap bytes
    board = ((board >> 16) & v2) | ((board & v2) << 16); // Swap half-words
    board = (board >> 32) | (board << 32);               // Swap words

    return board;
}

// ChessPosition createChessPosition(const Board &board,
//                                   const BoardStatus &status,
//                                   const uint64_t &epTarget)
// {
//     // Convert the en passant target into a single-bit bitboard if it's valid
//     uint64_t epBitboard = 0ULL;
//     if (status.HasEPPawn && epTarget.squareIndex >= 0 && epTarget.squareIndex < 64)
//     {
//         epBitboard = 1ULL << epTarget.squareIndex;
//     }

//     ChessPosition position = {
//         // White pieces
//         board.WPawn,
//         board.WKnight,
//         board.WBishop,
//         board.WRook,
//         board.WQueen,
//         board.WKing,

//         // Black pieces
//         board.BPawn,
//         board.BKnight,
//         board.BBishop,
//         board.BRook,
//         board.BQueen,
//         board.BKing,

//         // En passant bitboard
//         epBitboard,

//         // White to move, from BoardStatus
//         status.WhiteMove,

//         // Castling
//         status.WCastleL,
//         status.WCastleR,
//         status.BCastleL,
//         status.BCastleR};

//     return position;
// }