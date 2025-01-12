#include "../include/move_gen.h"
#include "../include/evaluate.h"
// #include "../giga/Chess_Base.hpp"
#include "../giga/Gigantua.hpp"
// #include "../include/data_preparation.h"
// #include "../include/chessnet.h"
#include <iostream>
#include <limits>
#include <future>
// #include <unordered_map>

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
// float alpha_beta(ChessNet &model, const std::string &pos, int depth, float alpha, float beta, bool isWhite)
// {
//     if (depth == 0)
//     {
//         return evaluate(model, pos); // Pass model to evaluate function
//     }

//     std::vector<std::string> positions = generate_positions(pos, isWhite);
//     if (positions.empty())
//     {
//         return evaluate(model, pos); // Handle checkmate or stalemate
//     }

//     if (isWhite)
//     {
//         float max_eval = std::numeric_limits<float>::lowest(); // Use lowest for float
//         for (const std::string &new_pos : positions)
//         {
//             float eval = alpha_beta(model, new_pos, depth - 1, alpha, beta, false);
//             max_eval = std::max(max_eval, eval);
//             alpha = std::max(alpha, eval);
//             // std::cout << "alpha:" << alpha << std::endl;
//             if (beta <= alpha){
//                 std::cout << "depth: " << depth << std::endl;
//                 std::cout << "Cutoff!" << depth << std::endl;
//                 break; // Beta cutoff
//             }
//         }
//         return max_eval;
//     }
//     else
//     {
//         float min_eval = std::numeric_limits<float>::max(); // Use max for float
//         for (const std::string &new_pos : positions)
//         {
//             float eval = alpha_beta(model, new_pos, depth - 1, alpha, beta, true);
//             min_eval = std::min(min_eval, eval);
//             beta = std::min(beta, eval);
//             // std::cout << "beta:" << beta << std::endl;
//             if (beta <= alpha){
//                 std::cout << "depth: " << depth << std::endl;
//                 std::cout << "Cutoff!" << depth << std::endl;
//                 break; // Alpha cutoff
//             }
//         }
//         return min_eval;
//     }
// }

// Evaluate the given chess position (stub implementation).
// float evaluate(ChessNet &model, const std::string &pos)
// {
//     // Convert the FEN position to bitboards and then to a tensor
//     ChessData positionInBitboards = fenToBitboards(pos);
//     torch::Tensor positionINTensor = bitboardsToTensor(positionInBitboards.bitboards);
//     // Reshape input tensor to [1, 14, 8, 8] for batch processing
//     positionINTensor = positionINTensor.unsqueeze(0);

//     // Check if CUDA is available
//     if (torch::cuda::is_available()) {
//         positionINTensor = positionINTensor.to(torch::kCUDA);
//     }
//     // Forward pass
//     torch::Tensor output = model.forward(positionINTensor);

//     // Retrieve output as a float
//     float eval = output.item<float>();

//     // Print the evaluation and return
//     std::cout << pos << " eval zł: " << eval << std::endl;
//     return eval;
// }


// 2808.73 seconds.
// Search function to find the best move.
std::string search_best_move(ChessNet model, std::string pos, int depth, std::unordered_map<uint64_t, float> &evaluations_map)
{

    // change
    bool isWhite_var = isWhite(pos);
    float best_eval;
    if (isWhite_var)
    {
        best_eval = std::numeric_limits<float>::lowest();
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
        // float eval = alpha_beta(model, new_pos, depth - 1, std::numeric_limits<float>::min(), std::numeric_limits<float>::max(), !isWhite_var);
        std::cout << "Evaluating positionn: " << new_pos << std::endl;
        float eval = _PerfT(new_pos, depth - 1, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), model, evaluations_map);
        std::cout << "eval: " << eval << std::endl;
        if (isWhite_var)
        {
            // best_eval = std::max(best_eval, eval);
            if (eval >= 1) {return new_pos;}
            if (eval > best_eval)
            {
                best_eval = eval;
                best_move = new_pos;
            }
        }
        else
        {   
            if (eval <= -1) {return new_pos;}
            if (eval < best_eval)
            {
                best_eval = eval;
                best_move = new_pos;
            }
        }
    }
    std::cout << "najlepszy: " << best_move << std::endl;
    std::cout << "eval: " << best_eval << std::endl;
    return best_move;
}



// Time taken to find best move: 178.041 seconds. depth = 4
// std::string search_best_move(ChessNet &model, const std::string pos, int depth)
// {
//     bool isWhite_var = isWhite(pos);
//     float best_eval = isWhite_var ? std::numeric_limits<float>::min() : std::numeric_limits<float>::max();
//     std::string best_move = "";

//     // Generate all possible moves
//     std::vector<std::string> positions = generate_positions(pos, isWhite_var);

//     // Store futures for each asynchronous evaluation
//     std::vector<std::future<std::pair<float, std::string>>> futures;

//     for (const auto &new_pos : positions)
//     {
//         // Launch each evaluation in a separate asynchronous task
//         futures.push_back(std::async(std::launch::async, [&model, new_pos, depth, isWhite_var]() {
//             float eval = _PerfT(new_pos, depth - 1, std::numeric_limits<float>::min(), std::numeric_limits<float>::max(), model);
//             std::cout << "nodes: " << MoveReciever::nodes << std::endl;
//             return std::make_pair(eval, new_pos);
//         }));
//     }

//     // Collect the results and determine the best move
//     for (auto &fut : futures)
//     {
//         auto [eval, move] = fut.get();
//         if ((isWhite_var && eval > best_eval) || (!isWhite_var && eval < best_eval))
//         {
//             best_eval = eval;
//             best_move = move;
//         }
//     }

//     return best_move;
// }