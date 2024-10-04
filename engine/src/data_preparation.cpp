#include "../include/data_preparation.h"


// Function to convert en passant square to bitboard (corrected for full 180-degree rotation)
uint64_t EnPassantToBitboard(const std::string& FEN) {
    std::map<char, int> fileMap = {{'a', 7}, {'b', 6}, {'c', 5}, {'d', 4}, {'e', 3}, {'f', 2}, {'g', 1}, {'h', 0}};
    size_t spaceCount = 0;
    size_t enPassantPos = 0;

    // Locate the en passant field (5th field in the FEN string)
    for (size_t i = 0; i < FEN.size(); ++i) {
        if (FEN[i] == ' ') {
            ++spaceCount;
            if (spaceCount == 3) { // En passant field is the 4th part
                enPassantPos = i + 1;
                break;
            }
        }
    }

    // If there's no en passant square or it's not valid, return 0
    if (enPassantPos == 0 || FEN.substr(enPassantPos, 1) == "-") {
        return 0; // No en passant square
    }

    char file = FEN[enPassantPos];       // File character (a-h)
    char rank = FEN[enPassantPos + 1];   // Rank character (1-8)
    
    // Get the file and rank index from the FEN string
    int fileIndex = fileMap[file];       // Convert file to an index 0-7
    int rankIndex = rank - '1';          // Convert '1'-'8' to 0-7

    // Flip file and rank for 180-degree board rotation
    // int flippedFileIndex = 7 - fileIndex; // Flip the file (h -> a, g -> b, etc.)
    // int flippedRankIndex = 7 - rankIndex; // Flip the rank (8 -> 1, 7 -> 2, etc.)

    // Calculate the square index for the rotated board
    int squareIndex = rankIndex * 8 + fileIndex;
    
    // Return the bitboard with the bit set at the rotated square index
    return 1ull << squareIndex;
}


uint64_t FenToBmp(const std::string& FEN, char p) {
    uint64_t result = 0;
    int field = 63;
    for (size_t i = 0; i < FEN.size() && FEN[i] != ' '; ++i) {
        char c = FEN[i];
        uint64_t P = 1ull << field;
        switch (c) {
            case '/': 
                field += 1; 
                break;
            case '1': 
                break;
            case '2': 
                field -= 1; 
                break;
            case '3': 
                field -= 2; 
                break;
            case '4': 
                field -= 3; 
                break;
            case '5': 
                field -= 4; 
                break;
            case '6': 
                field -= 5; 
                break;
            case '7': 
                field -= 6; 
                break;
            case '8': 
                field -= 7; 
                break;
            default:
                if (c == p) result |= P;
        }
        field--;
    }
    return result;
}


uint64_t extractInfoFromFEN(const std::string& FEN) {
    size_t spaceCount = 0;
    size_t castlingPos = 0;
    size_t turnPos = 0;
    
    // Locate the castling rights and turn fields in the FEN string
    for (size_t i = 0; i < FEN.size(); ++i) {
        if (FEN[i] == ' ') {
            ++spaceCount;
            if (spaceCount == 1) {
                turnPos = i + 1;  // Turn info (2nd field)
            }
            if (spaceCount == 2) {
                castlingPos = i + 1;  // Castling rights (3rd field)
                break;
            }
        }
    }

    uint64_t info = 0;

    // Determine the turn (White or Black)
    if (FEN[turnPos] == 'b') {
        info |= (1ull << 4);  // Bit 4 represents whose turn it is (1 for Black)
    }

    // Parse the castling rights
    for (size_t i = castlingPos; FEN[i] != ' '; ++i) {
        switch (FEN[i]) {
            case 'K': info |= (1ull << 0); break;  // White kingside castling (Bit 0)
            case 'Q': info |= (1ull << 1); break;  // White queenside castling (Bit 1)
            case 'k': info |= (1ull << 2); break;  // Black kingside castling (Bit 2)
            case 'q': info |= (1ull << 3); break;  // Black queenside castling (Bit 3)
        }
    }

    return info;  // The first 5 bits will have the information, rest will be zeros
}

torch::Tensor bitboardsToTensor(const std::vector<std::vector<std::vector<int>>>& bitboards) {
    std::vector<torch::Tensor> channels;

    for (const auto& board : bitboards) {
        torch::Tensor board_tensor = torch::from_blob(const_cast<int*>(board[0].data()), {8, 8}, torch::kInt32).clone();
        channels.push_back(board_tensor.unsqueeze(0));
    }

    return torch::cat(channels, 0).to(torch::kFloat32);
}

std::vector<std::vector<int>> intToBitboard(uint64_t bitboard) {
    std::vector<std::vector<int>> board(8, std::vector<int>(8, 0));
    for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 8; ++col) {
            // Extract each bit and place it in the 8x8 matrix
            board[row][col] = (bitboard >> (row * 8 + col)) & 1;
        }
    }
    return board;
}

ChessData fenToBitboards(const std::string& FEN){
    ChessData bitboards;
    bitboards.bitboards.resize(14);

    std::vector<std::string> whitePieces = {"P", "N", "B", "R", "Q", "K"};
    std::vector<std::string> blackPieces = {"p", "n", "b", "r", "q", "k"};


    for (size_t i = 0; i < whitePieces.size(); ++i) {
        bitboards.bitboards[i] = intToBitboard(FenToBmp(FEN, whitePieces[i][0]));
        }

    // Bind the bitboard values for black pieces
    for (size_t i = 0; i < blackPieces.size(); ++i) {
        bitboards.bitboards[i + whitePieces.size()] = intToBitboard(FenToBmp(FEN, blackPieces[i][0]));
    }

    bitboards.bitboards[whitePieces.size() + blackPieces.size()] = intToBitboard(EnPassantToBitboard(FEN));
    
    bitboards.bitboards[whitePieces.size() + blackPieces.size() + 1] = intToBitboard(extractInfoFromFEN(FEN));

    return bitboards;
}
