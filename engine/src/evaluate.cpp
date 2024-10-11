#include "../include/move_gen.h"
#include "../include/evaluate.h"
// #include "../include/data_preparation.h"
// #include "../include/chessnet.h"
#include <iostream>
#include <limits>

bool isWhite(const std::string &fen)
{
    if (fen.find('w') != std::string::npos)
    {
        std::cout << "whites turn" << std::endl;
        return true;
    }
    else
    {
        std::cout << "blacks turn" << std::endl;
        return false;
    }
}

// Generate all legal positions from the current position (stub implementation).
std::vector<std::string> generate_positions(std::string pos, bool isWhite)
{
    Midnight::Position board{pos}; // Load the starting FEN

    // Generate moves depending on the turn
    if (isWhite)
    {
        Midnight::MoveList<Midnight::WHITE, Midnight::MoveGenerationType::ALL> move_list(board);

        std::vector<std::string> positions;
        for (const auto &move : move_list)
        {
            board.play<Midnight::WHITE>(move); // Play the move for white
            positions.push_back(board.fen());  // Store the FEN
            board.undo<Midnight::WHITE>(move); // Undo the move to restore the state
        }
        return positions;
    }
    else
    {
        Midnight::MoveList<Midnight::BLACK, Midnight::MoveGenerationType::ALL> move_list(board);

        std::vector<std::string> positions;
        for (const auto &move : move_list)
        {
            board.play<Midnight::BLACK>(move); // Play the move for black
            positions.push_back(board.fen());  // Store the FEN
            board.undo<Midnight::BLACK>(move); // Undo the move to restore the state
        }
        return positions;
    }
}

// Alpha-beta pruning algorithm with neural network evaluation
float alpha_beta(ChessNet &model, const std::string &pos, int depth, float alpha, float beta, bool isWhite)
{
    if (depth == 0)
    {
        return evaluate(model, pos); // Pass model to evaluate function
    }

    std::vector<std::string> positions = generate_positions(pos, isWhite);
    if (positions.empty())
    {
        return evaluate(model, pos); // Handle checkmate or stalemate
    }

    if (isWhite)
    {
        float max_eval = std::numeric_limits<float>::lowest(); // Use lowest for float
        for (const std::string &new_pos : positions)
        {
            float eval = alpha_beta(model, new_pos, depth - 1, alpha, beta, false);
            max_eval = std::max(max_eval, eval);
            alpha = std::max(alpha, eval);
            // std::cout << "alpha:" << alpha << std::endl;
            if (beta <= alpha)
                break; // Beta cutoff
        }
        return max_eval;
    }
    else
    {
        float min_eval = std::numeric_limits<float>::max(); // Use max for float
        for (const std::string &new_pos : positions)
        {
            float eval = alpha_beta(model, new_pos, depth - 1, alpha, beta, true);
            min_eval = std::min(min_eval, eval);
            beta = std::min(beta, eval);
            // std::cout << "beta:" << beta << std::endl;
            if (beta <= alpha)
                break; // Alpha cutoff
        }
        return min_eval;
    }
}

// Evaluate the given chess position (stub implementation).
float evaluate(ChessNet &model, const std::string &pos)
{
    // ChessData positionInBitboards;
    // positionInBitboards.bitboards.resize(14);
    // positionInBitboards.bitboards = fenToBitboards(pos);

    ChessData positionInBitboards = fenToBitboards(pos);
    torch::Tensor positionINTensor = bitboardsToTensor(positionInBitboards.bitboards);
    // Reshape input tensor to [1, 14, 8, 8] for batch processing
    positionINTensor = positionINTensor.unsqueeze(0);
    torch::Tensor output = model.forward(positionINTensor);

    std::cout << pos << "eval:" << output.item<float>() << std::endl;
    // Extract scalar value and return as float
    return output.item<float>();
}

// Search function to find the best move.
std::string search_best_move(ChessNet &model, std::string pos, int depth)
{

    // change
    bool isWhite_var = isWhite(pos);
    float best_eval;
    if (isWhite_var)
    {
        best_eval = std::numeric_limits<float>::min();
    }
    else
    {
        best_eval = std::numeric_limits<float>::max();
    }
    std::string best_move = "dupa blada";
    std::vector<std::string> positions = generate_positions(pos, isWhite_var);

    // std::cout << "1" << isWhite_var << std::endl;
    // std::cout << "2" << isWhite_var << std::endl;
    // std::cout << "pos " << positions[1] << std::endl;
    for (const auto &new_pos : positions)
    {
        float eval = alpha_beta(model, new_pos, depth - 1, std::numeric_limits<float>::min(), std::numeric_limits<float>::max(), !isWhite_var);
        if (isWhite_var)
        {
            if (eval > best_eval)
            {
                best_eval = eval;
                best_move = new_pos;
            }
        }
        else
        {
            if (eval < best_eval)
            {
                best_eval = eval;
                best_move = new_pos;
            }
        }
    }
    std::cout << "najlepszy" << best_move << std::endl;
    return best_move;
}