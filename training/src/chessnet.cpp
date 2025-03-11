#include "../include/chessnet.h"
#include <iostream>

std::vector<std::vector<int>> intToBitboard(uint64_t bitboard, int value)
{
    std::vector<std::vector<int>> board(8, std::vector<int>(8, 0));
    for (int row = 0; row < 8; row++)
    {
        for (int col = 0; col < 8; col++)
        {
            // Extract each bit and place it in the 8x8 matrix
            board[row][col] = ((bitboard >> (row * 8 + col)) & 1) ? value : 0;
        }
    }
    return board;
}

uint64_t flipVertical(uint64_t board, bool isWhite)
{
    if (isWhite)
    {
        return board; // Do nothing for white pieces
    }

    return ((board << 56) & 0xFF00000000000000ULL) |
           ((board << 40) & 0x00FF000000000000ULL) |
           ((board << 24) & 0x0000FF0000000000ULL) |
           ((board << 8) & 0x000000FF00000000ULL) |
           ((board >> 8) & 0x00000000FF000000ULL) |
           ((board >> 24) & 0x0000000000FF0000ULL) |
           ((board >> 40) & 0x000000000000FF00ULL) |
           ((board >> 56) & 0x00000000000000FFULL);
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
    // x = fc3_bn(fc3(x));

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
                torch::kReLU);
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
                torch::kReLU);
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
    auto wPawn = intToBitboard(position.WPawn, 1);
    auto wKnight = intToBitboard(position.WKnight, 3);
    auto wBishop = intToBitboard(position.WBishop, 3);
    auto wRook = intToBitboard(position.WRook, 5);
    auto wQueen = intToBitboard(position.WQueen, 9);
    auto wKing = intToBitboard(position.WKing, 10);

    auto bPawn = intToBitboard(position.BPawn, -1);
    auto bKnight = intToBitboard(position.BKnight, -3);
    auto bBishop = intToBitboard(position.BBishop, -3);
    auto bRook = intToBitboard(position.BRook, -5);
    auto bQueen = intToBitboard(position.BQueen, -9);
    auto bKing = intToBitboard(position.BKing, -10);

    auto enPassant = intToBitboard(position.EnPassant, 1); // neutral => 0 or 1

    // We collect them into a single container for easier iteration
    // Each element is an 8x8 matrix (std::vector<std::vector<int>>)
    std::vector<std::vector<std::vector<int>>> allBoards = {
        wPawn, wKnight, wBishop, wRook, wQueen, wKing,
        bPawn, bKnight, bBishop, bRook, bQueen, bKing,
        enPassant};

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
                torch::kReLU);
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

    // Process "my"
    process(position.WPawn,false);
    process(position.WKnight,false);
    process(position.WBishop,false);
    process(position.WRook,false);
    process(position.WQueen,false);
    process(position.WKing,false);

    // Process enemy pieces
    process(position.BPawn, true);
    process(position.BKnight, true);
    process(position.BBishop, true);
    process(position.BRook, true);
    process(position.BQueen, true);
    process(position.BKing, true);

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

uint32_t calculateHalfKPIndex(uint64_t kingSquare, uint64_t pieceSquare,
                              PieceType pieceType, bool isWhite)
{
    // Map the piece type to the correct range for White or Black
    // 6 piece types for White, 6 for Black
    uint32_t pieceOffset = isWhite ? 0 : 6;

    // Calculate the relative square of the piece to the king
    // (Example logic; adapt if your board representation differs)
    uint32_t relativeSquare = static_cast<uint32_t>(pieceSquare - kingSquare);

    // Combine king square, piece type, and relative square into a feature index
    // The factor "64 * 10" is just an example; ensure it matches your indexing
    return (static_cast<uint32_t>(kingSquare) * 64 * 10) + (relativeSquare * 10) + (pieceOffset + static_cast<uint32_t>(pieceType));
}

// ------------------------------------------------------------
// NNUEHalfKPImpl Definition
// ------------------------------------------------------------
NNUEHalfKPImpl::NNUEHalfKPImpl()
{
    // Example "hard-coded" defaults
    int defaultInput = 41024; // e.g., 64 squares * piece types, etc.
    int defaultHidden = 256;
    int defaultOutput = 1;

    fc1 = register_module("fc1", torch::nn::Linear(defaultInput, defaultHidden));
    fc2 = register_module("fc2", torch::nn::Linear(defaultHidden, defaultHidden));
    fc3 = register_module("fc3", torch::nn::Linear(defaultHidden, defaultOutput));

    bn1 = register_module("bn1", torch::nn::BatchNorm1d(defaultHidden));
    bn2 = register_module("bn2", torch::nn::BatchNorm1d(defaultHidden));
    bn3 = register_module("bn3", torch::nn::BatchNorm1d(defaultOutput));

    initialize_weights();
}

// ------------------------------------------------------------
// Forward pass
// ------------------------------------------------------------
torch::Tensor NNUEHalfKPImpl::forward(torch::Tensor x)
{
    // If shape is [inputSize], unsqueeze to [1, inputSize]
    if (x.dim() == 1)
    {
        x = x.unsqueeze(0);
    }

    // std::cout << x[0] << std::endl;

    // fc1 -> bn1 -> relu
    x = fc1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);

    // fc2 -> bn2 -> relu
    x = fc2->forward(x);
    x = bn2->forward(x);
    x = torch::relu(x);

    // fc3 -> bn3 -> tanh
    x = fc3->forward(x);
    x = bn3->forward(x);
    x = torch::tanh(x);

    return x;
}

// ------------------------------------------------------------
// Initialize weights
// ------------------------------------------------------------
void NNUEHalfKPImpl::initialize_weights()
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
        // For each BatchNorm1d layer
        else if (auto *bn1d = dynamic_cast<torch::nn::BatchNorm1dImpl *>(module.get()))
        {
            torch::nn::init::ones_(bn1d->weight);
            torch::nn::init::zeros_(bn1d->bias);
        }
    }
}

// ------------------------------------------------------------
// toTensor: Convert a ChessPosition to a dense tensor
// ------------------------------------------------------------
torch::Tensor NNUEHalfKPImpl::toTensor(const ChessPosition &position)
{
    // 1. Gather indices as int64_t directly.
    std::vector<int64_t> featureIndices = createHalfKPInputVector(position);

    // 2. Create a 1D int64 tensor from the data.
    //    Use .clone() so that the tensor owns the memory
    auto indices = torch::from_blob(
        featureIndices.data(),
        {static_cast<long>(featureIndices.size())},
        torch::kInt64
    ).clone();

    // 3. Create the values tensor (float32) with the same length.
    auto values = torch::ones(
        {static_cast<long>(featureIndices.size())},
        torch::TensorOptions().dtype(torch::kFloat32)
    );

    // 4. Build the sparse COO tensor.
    //    - indices must have shape [1, N] or [2, N], so we do unsqueeze(0) 
    //      to get [1, N].
    auto sparseTensor = torch::sparse_coo_tensor(
        indices.unsqueeze(0), // shape [1, N]
        values,               // shape [N]
        {41024}               // total size of the 1D feature space
    );

    // 5. Convert to dense.
    auto denseTensor = sparseTensor.to_dense();

    // 6. (Optional) move to GPU if available.
    if (torch::cuda::is_available()) {
        denseTensor = denseTensor.to(torch::kCUDA);
    }

    // 7. Return your final dense tensor.
    return denseTensor;
}



// ------------------------------------------------------------
// createHalfKPInputVector: Gather feature indices
// ------------------------------------------------------------
std::vector<int64_t> NNUEHalfKPImpl::createHalfKPInputVector(const ChessPosition &position)
{
    // We ultimately want a std::vector<int64_t> to return
    std::vector<int64_t> featureIndices;
    featureIndices.reserve(256); // or whatever approximate upper bound you want

    // Example usage of uint64_t for bitboards
    uint64_t whiteKingSquare = __builtin_ctzll(position.WKing);
    uint64_t blackKingSquare = __builtin_ctzll(position.BKing);

    // Helper lambda to add features for a given side
    auto addFeatures = [&](uint64_t kingSquare, const uint64_t *pieces, bool isWhite) {
        for (int pieceType = PAWN; pieceType <= KING; ++pieceType)
        {
            uint64_t pieceBitboard = pieces[pieceType];
            while (pieceBitboard)
            {
                uint64_t pieceSquare = __builtin_ctzll(pieceBitboard);
                
                // calculateHalfKPIndex returns uint32_t, 
                // but we cast to int64_t before storing
                int64_t idx = static_cast<int64_t>(calculateHalfKPIndex(
                    kingSquare,
                    pieceSquare,
                    static_cast<PieceType>(pieceType),
                    isWhite
                ));
                
                featureIndices.push_back(idx);

                // Clear the LSB
                pieceBitboard &= (pieceBitboard - 1);
            }
        }
    };

    // Arrays of piece bitboards for White/Black
    const uint64_t whitePieces[] = {
        position.WPawn, position.WKnight, position.WBishop,
        position.WRook, position.WQueen, position.WKing
    };
    const uint64_t blackPieces[] = {
        position.BPawn, position.BKnight, position.BBishop,
        position.BRook, position.BQueen, position.BKing
    };

    // Add features for White
    addFeatures(whiteKingSquare, whitePieces, true);

    // Add features for Black
    addFeatures(blackKingSquare, blackPieces, false);

    // Return the int64_t vector
    return featureIndices;
}




// ------------------------------------------
// ChessNetConv (Convolution-based)
// ------------------------------------------
ChessNetConv2Impl::ChessNetConv2Impl()
    : conv1(torch::nn::Conv2dOptions(13, 64, 3).stride(1).padding(1)),
      conv2(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),

      bn1(64), bn2(128),

      conv_flat_bn(128 * 8 * 8),

      fc1(128 * 8 * 8, 512),
      fc1_bn(512),
      fc2(512, 128),
      fc2_bn(128),
      fc3_bn(1),
      fc3(128, 1)
{
    // Register modules
    register_module("conv1", conv1);
    register_module("conv2", conv2);

    register_module("bn1", bn1);
    register_module("bn2", bn2);

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

torch::Tensor ChessNetConv2Impl::forward(torch::Tensor x)
{
    // Convolution + BN + ReLU
    x = torch::relu(bn1(conv1(x)));
    x = torch::relu(bn2(conv2(x)));

    // Flatten from [batch_size, 512, 8, 8] to [batch_size, 512*8*8]
    x = x.view({-1, 128 * 8 * 8});

    // BatchNorm on flattened conv output
    x = conv_flat_bn(x);

    // FC1 -> BN -> ReLU
    x = torch::relu(fc1_bn(fc1(x)));

    // FC2 -> BN -> ReLU
    x = torch::relu(fc2_bn(fc2(x)));

    // FC3 -> BN -> Tanh => output in [-1, 1]
    x = torch::tanh(fc3_bn(fc3(x)));
    // x = fc3_bn(fc3(x));

    // Debug: if batch is large, print stats
    if (x.size(0) > 100)
    {
        std::cout << "output avg: " << x.mean().item().toFloat() << std::endl;
        std::cout << "output range: " << x.min().item().toFloat()
                  << " to " << x.max().item().toFloat() << std::endl;
    }

    return x;
}

void ChessNetConv2Impl::initialize_weights()
{
    for (auto &module : modules(/*include_self=*/false))
    {
        if (auto *conv = dynamic_cast<torch::nn::Conv2dImpl *>(module.get()))
        {
            torch::nn::init::kaiming_uniform_(
                conv->weight, /*a=*/0.01,
                torch::kFanIn,
                torch::kReLU);
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
                torch::kReLU);
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

torch::Tensor ChessNetConv2Impl::toTensor(const ChessPosition &position)
{

    // Convert each 64-bit bitboard into an 8x8 2D vector of ints
    auto wPawn = intToBitboard(position.WPawn, 1);
    auto wKnight = intToBitboard(position.WKnight, 3);
    auto wBishop = intToBitboard(position.WBishop, 3);
    auto wRook = intToBitboard(position.WRook, 5);
    auto wQueen = intToBitboard(position.WQueen, 9);
    auto wKing = intToBitboard(position.WKing, 10);

    auto bPawn = intToBitboard(position.BPawn, -1);
    auto bKnight = intToBitboard(position.BKnight, -3);
    auto bBishop = intToBitboard(position.BBishop, -3);
    auto bRook = intToBitboard(position.BRook, -5);
    auto bQueen = intToBitboard(position.BQueen, -9);
    auto bKing = intToBitboard(position.BKing, -10);

    auto enPassant = intToBitboard(position.EnPassant, 1); // neutral => 0 or 1

    // We collect them into a single container for easier iteration
    // Each element is an 8x8 matrix (std::vector<std::vector<int>>)
    std::vector<std::vector<std::vector<int>>> allBoards = {
        wPawn, wKnight, wBishop, wRook, wQueen, wKing,
        bPawn, bKnight, bBishop, bRook, bQueen, bKing,
        enPassant};

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