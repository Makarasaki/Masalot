#ifndef EVALUATE_H
#define EVALUATE_H

#include <torch/torch.h>
#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>
#include "../../training/include/chessnet.h"
#include "../include/data_preparation.h"


std::vector<std::string> generate_positions(std::string pos, bool isWhite);

float alpha_beta(ChessNet& model, const std::string& pos, int depth, float alpha, float beta, bool isWhite);

float evaluate(ChessNet model, const std::string& pos);

bool isWhite(const std::string& fen);

// std::string search_best_move(ChessNet &model, std::string &pos, int depth, std::unordered_map<uint64_t, float> &evaluations_map, const std::unordered_set<std::string> &previous_positions);

std::string search_best_move(
    ChessNet &model,
    const std::string &pos,
    int depth,
    std::unordered_map<uint64_t, float> &evaluations_map,
    std::unordered_set<std::string> &previous_positions // note: pass by reference
);

std::string stripFen(const std::string &fen);

int countBoardPoints(const std::string& fen);

#endif  // EVALUATE_H