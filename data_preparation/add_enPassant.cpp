#include <iostream>
#include <string>
#include <sqlite3.h>
#include <map>

// Function to check if a column exists in a table
bool columnExists(sqlite3* db, const std::string& tableName, const std::string& columnName) {
    std::string query = "PRAGMA table_info(" + tableName + ");";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, 0) != SQLITE_OK) {
        std::cerr << "Error preparing statement: " << sqlite3_errmsg(db) << std::endl;
        return false;
    }

    bool exists = false;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        std::string name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        if (name == columnName) {
            exists = true;
            break;
        }
    }

    sqlite3_finalize(stmt);
    return exists;
}

// Function to add the en passant column if it doesn't exist
void addEnPassantColumn(sqlite3* db, const std::string& tableName) {
    if (!columnExists(db, tableName, "en_passant_bitboard")) {
        std::string query = "ALTER TABLE " + tableName + " ADD COLUMN en_passant_bitboard INTEGER";
        if (sqlite3_exec(db, query.c_str(), 0, 0, 0) != SQLITE_OK) {
            std::cerr << "Error adding en passant column: " << sqlite3_errmsg(db) << std::endl;
        }
    }
}

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

// Function to update the en passant bitboard in the database for each FEN
void updateEnPassantBitboard(sqlite3* db, const std::string& tableName) {
    sqlite3_stmt* stmt;
    std::string select_query = "SELECT fen FROM " + tableName;
    std::string update_query = "UPDATE " + tableName + " SET en_passant_bitboard = ? WHERE fen = ?";

    if (sqlite3_prepare_v2(db, select_query.c_str(), -1, &stmt, 0) != SQLITE_OK) {
        std::cerr << "Error preparing select statement: " << sqlite3_errmsg(db) << std::endl;
        return;
    }

    sqlite3_stmt* update_stmt;
    if (sqlite3_prepare_v2(db, update_query.c_str(), -1, &update_stmt, 0) != SQLITE_OK) {
        std::cerr << "Error preparing update statement: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_finalize(stmt);
        return;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        std::string fen = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        // std::string evaluation = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));

        uint64_t enPassantBitboard = EnPassantToBitboard(fen);
        sqlite3_bind_int64(update_stmt, 1, enPassantBitboard);
        sqlite3_bind_text(update_stmt, 2, fen.c_str(), -1, SQLITE_STATIC);
        // sqlite3_bind_text(update_stmt, 3, evaluation.c_str(), -1, SQLITE_STATIC);

        if (sqlite3_step(update_stmt) != SQLITE_DONE) {
            std::cerr << "Error updating row with FEN " << fen << ": " << sqlite3_errmsg(db) << std::endl;
        }

        sqlite3_reset(update_stmt);
    }

    sqlite3_finalize(stmt);
    sqlite3_finalize(update_stmt);
}

int main() {
    std::string tableName = "big_db_100_000";
    sqlite3* db;
    int rc = sqlite3_open("../chess_evals.db", &db);
    if (rc) {
        std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
        return 1;
    }

    // Add en passant column if it doesn't exist
    addEnPassantColumn(db, tableName);

    // Update en passant bitboards for each row
    updateEnPassantBitboard(db, tableName);

    sqlite3_close(db);
    return 0;
}
