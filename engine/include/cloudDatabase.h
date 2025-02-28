#ifndef CLOUDDATABASE_H
#define CLOUDDATABASE_H

#include <string>

// Declaration of the function implemented in cloudDatabase.cpp
// Takes a FEN string as input, returns the best move.
std::string getBestMoveFromCDB(const std::string& fen);

#endif // CLOUDDATABASE_H