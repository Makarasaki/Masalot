#include "../include/move_gen.h"
#include "../include/evaluate.h"
#include "../include/cloudDatabase.h"
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
        // std::cout << "whites turn" << std::endl;
        return true;
    }
    else
    {
        // std::cout << "blacks turn" << std::endl;
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

/**
 * Returns the material points on the board given a FEN string,
 * counting pawns=1, knights=3, bishops=3, rooks=5, queens=9,
 * and ignoring kings, digits (empty squares), and slashes (/).
 */
int countBoardPoints(const std::string& fen)
{
    // 1) Extract piece placement (everything up to the first space).
    //    A typical FEN is "piece_placement side_to_move castling ...", 
    //    so we'll find the substring before the space:
    std::size_t spacePos = fen.find(' ');
    std::string placement = (spacePos == std::string::npos)
                            ? fen
                            : fen.substr(0, spacePos);

    int totalPoints = 0;

    // 2) Iterate over each character in the piece placement.
    for (char c : placement)
    {
        switch (c)
        {
            case 'P': case 'p': totalPoints += 1; break;  // Pawn
            case 'N': case 'n': totalPoints += 3; break;  // Knight
            case 'B': case 'b': totalPoints += 3; break;  // Bishop
            case 'R': case 'r': totalPoints += 5; break;  // Rook
            case 'Q': case 'q': totalPoints += 9; break;  // Queen
            // K/k (king) => 0, digits => empty squares, '/' => rank separator
            default:
                // do nothing for K/k, digits, slashes, etc.
                break;
        }
    }

    return totalPoints;
}

std::string stripFen(const std::string &fen) {
    std::istringstream fenStream(fen);
    std::string board, turn, castling, enPassant;
    std::string halfmoveClock, fullmoveNumber;

    // Read the FEN components
    fenStream >> board >> turn >> castling >> enPassant >> halfmoveClock >> fullmoveNumber;

    // Construct and return the stripped FEN
    return board + " " + turn + " " + castling + " " + enPassant;
}

// 2808.73 seconds.
// Search function to find the best move.
// std::string search_best_move(ChessNet model, std::string pos, int depth, std::unordered_map<uint64_t, float> &evaluations_map)
// {
//     std::string response = getBestMoveFromCDB(pos);
//     std::cout << "response from database: " << response << std::endl;
//     if(!(response == "nobestmove"))
//     {
//         return response;
//     }
//     int goDeeperTreshold = 20;
//     if (countBoardPoints(pos) < goDeeperTreshold){
//         std::cout << "Few pieces on the board, searching deeper" << std::endl;
//         goDeeperTreshold += 2;
//     }
//     // change
//     bool isWhite_var = isWhite(pos);
//     float best_eval;
//     if (isWhite_var)
//     {
//         best_eval = std::numeric_limits<float>::lowest();
//     }
//     else
//     {
//         best_eval = std::numeric_limits<float>::max();
//     }
//     std::string best_move = "dupa blada";
//     std::vector<std::string> positions = generate_positions(pos, isWhite_var);
//     int sumOfNodes = 0;
//     // std::cout << "1" << isWhite_var << std::endl;
//     // std::cout << "2" << isWhite_var << std::endl;
//     // std::cout << "pos " << positions[1] << std::endl;

//     for (const auto &new_pos : positions)
//     {
//         // float eval = alpha_beta(model, new_pos, depth - 1, std::numeric_limits<float>::min(), std::numeric_limits<float>::max(), !isWhite_var);
//         std::cout << "Evaluating positionn: " << new_pos << std::endl;
//         float eval = _PerfT(new_pos, depth - 1, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), model, evaluations_map);
//         std::cout << "nodes: " << MoveReciever::nodes << std::endl;
//         std::cout << "eval: " << eval << std::endl;
//         sumOfNodes += MoveReciever::nodes;
//         if (isWhite_var)
//         {
//             // best_eval = std::max(best_eval, eval);
//             if (eval >= 1) {return new_pos;}
//             if (eval > best_eval)
//             {
//                 best_eval = eval;
//                 best_move = new_pos;
//             }
//         }
//         else
//         {   
//             if (eval <= -1) {return new_pos;}
//             if (eval < best_eval)
//             {
//                 best_eval = eval;
//                 best_move = new_pos;
//             }
//         }
//     }
//     std::cout << "Best Move: " << best_move << std::endl;
//     std::cout << "eval: " << best_eval << std::endl;
//     std::cout << "Number of positions evaluated: " << sumOfNodes << std::endl;
//     return best_move;
// }


/**
 * @brief Finds the best move for a given position using a alpha-beta pruning algorithm with neural network as evaluation function.
 *        Avoids moves that would lead to repetition or allow the opponent 
 *        to immediately repeat. If the best "safe" move is losing, 
 *        return absolute best move found.
 *
 * @param model              Your chess engine model neural network
 * @param pos                Current position in FEN format
 * @param depth              Search depth
 * @param evaluations_map    Cache of evaluations keyed by Zobrist hash
 * @param previous_positions A set of positions that have already occurred
 * @return                   The chosen best move
 */
std::string search_best_move(
    ChessNet &model,
    const std::string &pos,
    int depth,
    std::unordered_map<uint64_t, float> &evaluations_map,
    std::unordered_set<std::string> &previous_positions // note: pass by reference
)
{
    // 0. Mark this position as visited
    previous_positions.insert(stripFen(pos));

    std::cout << "Number of previous positions: " << previous_positions.size() << std::endl;
    std::cout << "Number evaluated positions: " << evaluations_map.size() << std::endl;

    // 1. Check if there's a best move Chess Database
    std::string response = getBestMoveFromCDB(pos);
    std::cout << "response from database: " << response << std::endl;
    if (response != "nobestmove")
    {
        return response;
    }

    // 2. Possibly adjust search depth if the board is nearing endgame
    int goDeeperThreshold1 = 30;
    int goDeeperThreshold2 = 20;
    int goDeeperThreshold3 = 10;
    int boardPoints = countBoardPoints(pos);
    if (boardPoints < goDeeperThreshold1) {
        std::cout << "Few pieces on the board, searching deeper, treshhold 30" << std::endl;
        depth += 2; 
    }else if (boardPoints < goDeeperThreshold2)
    {
        std::cout << "Few pieces on the board, searching deeper, treshhold 20" << std::endl;
        depth += 3; 
    }else if (boardPoints < goDeeperThreshold3)
    {
        std::cout << "Few pieces on the board, searching deeper, treshhold 10" << std::endl;
        depth += 4; 
    }
    

    // 3. Determine whose turn it is
    bool isWhiteTurn = isWhite(pos);

    // 4. Generate all next positions (candidate moves)
    std::vector<std::string> next_positions = generate_positions(pos, isWhiteTurn);

    if (next_positions.empty())
    {
        std::cout << "No moves available - returning dummy move.\n";
        return "No moves available"; 
    }

    // 5. Evaluate each candidate position
    std::vector<std::pair<float, std::string>> evaluations; 
    evaluations.reserve(next_positions.size());

    int sumOfNodes = 0;

    for (const auto &new_pos : next_positions)
    {
        std::cout << "Evaluating position: " << new_pos << std::endl;

        float eval = _PerfT(new_pos,
                            depth - 1,
                            std::numeric_limits<float>::lowest(),
                            std::numeric_limits<float>::max(),
                            model,
                            evaluations_map);

        std::cout << "nodes: " << MoveReciever::nodes << std::endl;
        std::cout << "eval: " << eval << std::endl;

        sumOfNodes += MoveReciever::nodes;

        if(eval > 1 and isWhiteTurn)
        {
            std::cout << "Early return, found winning line for Whites" << std::endl;
            previous_positions.insert(stripFen(new_pos));
            std::cout << "Chosen Move: " << new_pos << std::endl;
            std::cout << "eval: " << eval << std::endl;
            std::cout << "Positions (nodes) evaluated: " << sumOfNodes << std::endl;
            return new_pos;
        }
        else if (eval < -1 and !isWhiteTurn)
        {
            std::cout << "Early return, found winning line for Blacks" << std::endl;
            previous_positions.insert(stripFen(new_pos));
            std::cout << "Chosen Move: " << new_pos << std::endl;
            std::cout << "eval: " << eval << std::endl;
            std::cout << "Positions (nodes) evaluated: " << sumOfNodes << std::endl;
            return new_pos;
        }
        

        // Collect the (eval, position) pair
        evaluations.emplace_back(eval, new_pos);
    }

    // 6. Sort the moves:
    //    - White to move => descending (higher eval is better)
    //    - Black to move => ascending (lower eval is better)
    if (isWhiteTurn)
    {
        std::sort(evaluations.begin(), evaluations.end(),
                  [](auto &a, auto &b) {
                      return a.first > b.first; 
                  });
    }
    else
    {
        std::sort(evaluations.begin(), evaluations.end(),
                  [](auto &a, auto &b) {
                      return a.first < b.first;
                  });
    }

    // 7. Identify the best move overall (ignoring repetition).
    float best_eval_overall = evaluations[0].first;
    std::string best_move_overall = evaluations[0].second;

    // Helper lambda: checks if a move is "safe" from repetition
    // i.e., the resulting position is not in previous_positions,
    // AND the opponent cannot immediately repeat from that position.
    auto isSafeMove = [&](const std::string &candidate_pos) {
        // If this new position is already in previous_positions => repetition risk
        if (previous_positions.find(stripFen(candidate_pos)) != previous_positions.end()) {
            std::cout << "repetition risk detected, case 1" << std::endl;
            return false;
        }

        // Next side to move
        bool nextSideWhite = isWhite(candidate_pos);
        // Generate the opponent's possible replies
        auto opp_positions = generate_positions(candidate_pos, nextSideWhite);

        // If *any* of those is already in previous_positions => opponent can force repetition
        for (const auto &opp_pos : opp_positions)
        {
            if (previous_positions.find(stripFen(opp_pos)) != previous_positions.end())
            {   
                std::cout << "repetition risk detected, case 2" << std::endl;
                return false;
            }
        }
        return true;
    };

    // 8. Gather "safe" moves (which won't cause or allow immediate repetition)
    std::vector<std::pair<float, std::string>> safe_moves;
    for (auto &ev : evaluations) {
        if (isSafeMove(ev.second)) {
            safe_moves.push_back(ev);
        }
    }

    float chosen_eval;
    std::string chosen_move;

    if (!safe_moves.empty())
    {
        if (isWhiteTurn)
        {
            std::sort(safe_moves.begin(), safe_moves.end(),
                      [](auto &a, auto &b) {
                          return a.first > b.first;
                      });
        }
        else
        {
            std::sort(safe_moves.begin(), safe_moves.end(),
                      [](auto &a, auto &b) {
                          return a.first < b.first;
                      });
        }

        // Best safe move
        chosen_eval = safe_moves[0].first;
        chosen_move = safe_moves[0].second;

        // 9. If the best safe move is "losing" by your threshold,
        //    revert to the overall best move.
        //    (Assuming your evaluation is from White's perspective.)
        //
        //    - White wants eval >= +0 to be "winning."
        //    - If White sees < 0, it's not good enough => revert to best overall.
        //    - If Black is to move and sees eval > 0, that means White is +0 => 
        //      black is losing => revert to best overall.
        bool losingForWhite = (isWhiteTurn && (chosen_eval < 0.0f));
        bool losingForBlack = (!isWhiteTurn && (chosen_eval > 0.0f));

        if (losingForWhite || losingForBlack)
        {
            std::cout << "Best safe move is below threshold => reverting to best overall.\n";
            chosen_eval = best_eval_overall;
            chosen_move = best_move_overall;
        }
    }
    else
    {
        // No safe moves exist => pick the best overall
        chosen_eval = best_eval_overall;
        chosen_move = best_move_overall;
    }

    std::cout << "Chosen Move: " << chosen_move << std::endl;
    std::cout << "eval: " << chosen_eval << std::endl;
    std::cout << "Positions (nodes) evaluated: " << sumOfNodes << std::endl;

    previous_positions.insert(stripFen(chosen_move));

    return chosen_move;
}















// Run Pararell:
// std::string search_best_move(ChessNet model, std::string pos, int depth, std::unordered_map<uint64_t, float> &evaluations_map)
// {
// // 1. Possibly check an opening database
// std::string response = getBestMoveFromCDB(pos);
// std::cout << "response from database: " << response << std::endl;
// // if (response != "nobestmove") {
// //     return response;
// // }

// // 2. Determine if it's White to move
// bool isWhite_var = isWhite(pos);

// // 3. Initialize best_eval
// float best_eval = isWhite_var
// ? std::numeric_limits<float>::lowest()
// : std::numeric_limits<float>::max();

// std::string best_move = "NO_MOVE_FOUND";
// std::vector<std::string> positions = generate_positions(pos, isWhite_var);

// // 4. Launch parallel tasks for each new position
// std::vector<std::future<std::pair<std::string, float>>> tasks;
// tasks.reserve(positions.size());

// for (const auto &new_pos : positions)
// {
// // Capture new_pos by value in the lambda, and pass references to everything else as needed
// tasks.push_back(std::async(std::launch::async, [=, &model, &evaluations_map]() {
// // Evaluate the position
// float eval = _PerfT(new_pos, depth - 1,
//        std::numeric_limits<float>::lowest(),
//        std::numeric_limits<float>::max(),
//        model, evaluations_map);
// // Return a pair: (move, eval)
// return std::make_pair(new_pos, eval);
// }));
// }

// // 5. Collect the results and choose the best move
// for (auto &task : tasks)
// {
// auto [pos_string, eval] = task.get();  // Wait for the thread to finish and get result
// std::cout << "pos: " << pos_string << ", eval: " << eval << "\n";

// if (isWhite_var) {
// if (eval >= 1.0f) {
// // As soon as we find a winning eval, we can return
// return pos_string;
// }
// if (eval > best_eval) {
// best_eval = eval;
// best_move = pos_string;
// }
// } else {
// if (eval <= -1.0f) {
// // As soon as we find a strongly losing eval (for White),
// // that might be winning for Black, so return
// return pos_string;
// }
// if (eval < best_eval) {
// best_eval = eval;
// best_move = pos_string;
// }
// }
// }

// std::cout << "Best move: " << best_move << "\n";
// std::cout << "eval: " << best_eval << "\n";
// return best_move;
// }