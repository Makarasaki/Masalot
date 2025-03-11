#include "../include/move_gen.h"
#include "../include/evaluate.h"
#include "../include/cloudDatabase.h"
#include "../giga/Gigantua.hpp"
#include <algorithm>  // For std::shuffle
#include <random>    
#include <iostream>
#include <limits>
#include <future>

bool isWhite(const std::string &fen)
{
    return fen.find('w') != std::string::npos;
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


bool isSafeMove(const std::string &candidate_pos, 
    const std::unordered_set<std::string> &previous_positions) {

// If this new position is already in previous_positions => repetition risk
if (previous_positions.find(stripFen(candidate_pos)) != previous_positions.end()) {
std::cout << "Repetition risk detected, case 1" << std::endl;
return false;
}

// Determine the next side to move
bool nextSideWhite = isWhite(candidate_pos);

// Generate the opponent's possible replies
auto opp_positions = generate_positions(candidate_pos, nextSideWhite);

// If *any* of those is already in previous_positions => opponent can force repetition
for (const auto &opp_pos : opp_positions) {
if (previous_positions.find(stripFen(opp_pos)) != previous_positions.end()) {   
std::cout << "Repetition risk detected, case 2" << std::endl;
return false;
}
}
return true;
}


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
bestMoveInfo search_best_move(
    ChessNet &model,
    const std::string &pos,
    int depth,
    std::unordered_map<uint64_t, float> &evaluations_map,
    std::unordered_set<std::string> &previous_positions // note: pass by reference
)
{

    bestMoveInfo chosen_move;
    chosen_move.nodes = 0;
    chosen_move.depth = 0;
    chosen_move.eval = 0;
    // 0. Mark this position as visited
    previous_positions.insert(stripFen(pos));

    std::cout << "Number of previous positions: " << previous_positions.size() << std::endl;
    std::cout << "Number evaluated positions: " << evaluations_map.size() << std::endl;

    // 1. Check if there's a best move Chess Database
    std::string response = getBestMoveFromCDB(pos);
    std::cout << "response from database: " << response << std::endl;
    if (response != "nobestmove")
    {
        chosen_move.move = response;
        return chosen_move;
    }

    // 2. Possibly adjust search depth if the board is nearing endgame
    int goDeeperThreshold1 = 20;
    int goDeeperThreshold2 = 10;
    int goDeeperThreshold3 = 5;
    int boardPoints = countBoardPoints(pos);
    if (boardPoints < goDeeperThreshold1) {
        std::cout << "Few pieces on the board, searching deeper, treshhold " << goDeeperThreshold1 << std::endl;
        depth += 1; 
    }else if (boardPoints < goDeeperThreshold2)
    {
        std::cout << "Few pieces on the board, searching deeper, treshhold " << goDeeperThreshold2 << std::endl;
        depth += 2; 
    }else if (boardPoints < goDeeperThreshold3)
    {
        std::cout << "Few pieces on the board, searching deeper, treshhold " << goDeeperThreshold3 << std::endl;
        depth += 3; 
    }
    chosen_move.depth = depth;
    

    // 3. Determine whose turn it is
    bool isWhiteTurn = isWhite(pos);

    // 4. Generate all next positions (candidate moves)
    std::vector<std::string> next_positions = generate_positions(pos, isWhiteTurn);


    if (next_positions.empty())
    {
        std::cout << "No moves available - returning dummy move.\n";
        chosen_move.move = "No moves available";
        return chosen_move;
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

        std::cout << "nodes: " << MoveReceiver::nodes << std::endl;
        std::cout << "eval: " << eval << std::endl;

        sumOfNodes += MoveReceiver::nodes;

        if(eval > 1 and isWhiteTurn and isSafeMove(new_pos, previous_positions))
        {
            std::cout << "Early return, found winning line for Whites" << std::endl;
            previous_positions.insert(stripFen(new_pos));
            std::cout << "Chosen Move: " << new_pos << std::endl;
            std::cout << "eval: " << eval << std::endl;
            std::cout << "Positions (nodes) evaluated: " << sumOfNodes << std::endl;
            chosen_move.move = new_pos;
            chosen_move.nodes = sumOfNodes;
            chosen_move.eval = eval;
            return chosen_move;
        }
        else if (eval < -1 and !isWhiteTurn and isSafeMove(new_pos, previous_positions))
        {
            std::cout << "Early return, found winning line for Blacks" << std::endl;
            previous_positions.insert(stripFen(new_pos));
            std::cout << "Chosen Move: " << new_pos << std::endl;
            std::cout << "eval: " << eval << std::endl;
            std::cout << "Positions (nodes) evaluated: " << sumOfNodes << std::endl;
            chosen_move.move = new_pos;
            chosen_move.nodes = sumOfNodes;
            chosen_move.eval = eval;
            return chosen_move;
        }
        

        // Collect the (eval, position) pair
        evaluations.emplace_back(eval, new_pos);
    }

    chosen_move.nodes = sumOfNodes;

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

    // 8. Gather "safe" moves (which won't cause or allow immediate repetition)
    std::vector<std::pair<float, std::string>> safe_moves;
    for (auto &ev : evaluations) {
        if (isSafeMove(ev.second, previous_positions)) {
            safe_moves.push_back(ev);
        }
    }

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
        chosen_move.eval = safe_moves[0].first;
        chosen_move.move = safe_moves[0].second;

        // 9. If the best safe move is "losing" by your threshold,
        //    revert to the overall best move.
        //    (Assuming your evaluation is from White's perspective.)
        //
        //    - White wants eval >= +0 to be "winning."
        //    - If White sees < 0, it's not good enough => revert to best overall.
        //    - If Black is to move and sees eval > 0, that means White is +0 => 
        //      black is losing => revert to best overall.
        bool losingForWhite = (isWhiteTurn && (chosen_move.eval < 0.0f));
        bool losingForBlack = (!isWhiteTurn && (chosen_move.eval > 0.0f));

        if (losingForWhite || losingForBlack)
        {
            std::cout << "Best safe move is below threshold => reverting to best overall.\n";
            chosen_move.eval = best_eval_overall;
            chosen_move.move = best_move_overall;
        }
    }
    else
    {
        // No safe moves exist => pick the best overall
        chosen_move.eval = best_eval_overall;
        chosen_move.move = best_move_overall;
    }

    std::cout << "Chosen Move: " << chosen_move.move << std::endl;
    std::cout << "eval: " << chosen_move.eval << std::endl;
    std::cout << "Positions (nodes) evaluated: " << sumOfNodes << std::endl;

    previous_positions.insert(stripFen(chosen_move.move));

    return chosen_move;
}
