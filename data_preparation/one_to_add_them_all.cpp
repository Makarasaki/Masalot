#include <iostream>
#include <string>
#include <sqlite3.h>
#include <bitset>
#include <vector>
#include <map>
#include <cstdint>

// g++ -o bin/one_to_add.exe one_to_add_them_all.cpp -lsqlite3

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

// Function to add columns for chess piece bitboards to a specific table
void addColumns(sqlite3* db, const std::string& tableName) {
    std::vector<std::string> whitePieces = {"P", "N", "B", "R", "Q", "K"};
    std::vector<std::string> blackPieces = {"p", "n", "b", "r", "q", "k"};

    // Add columns for white pieces
    for (const auto& piece : whitePieces) {
        std::string columnName = "w_" + piece + "_bitboard";
        if (!columnExists(db, tableName, columnName)) {
            std::string query = "ALTER TABLE " + tableName + " ADD COLUMN " + columnName + " INTEGER";
            if (sqlite3_exec(db, query.c_str(), 0, 0, 0) != SQLITE_OK) {
                std::cerr << "Error adding column " << columnName << ": " << sqlite3_errmsg(db) << std::endl;
            }
        }
    }

    // Add columns for black pieces
    for (const auto& piece : blackPieces) {
        std::string columnName = "b_" + piece + "_bitboard";
        if (!columnExists(db, tableName, columnName)) {
            std::string query = "ALTER TABLE " + tableName + " ADD COLUMN " + columnName + " INTEGER";
            if (sqlite3_exec(db, query.c_str(), 0, 0, 0) != SQLITE_OK) {
                std::cerr << "Error adding column " << columnName << ": " << sqlite3_errmsg(db) << std::endl;
            }
        }
    }
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

// Function to add the 'info' column if it doesn't exist
void addInfoColumn(sqlite3* db, const std::string& tableName) {
    if (!columnExists(db, tableName, "info")) {
        std::string query = "ALTER TABLE " + tableName + " ADD COLUMN info INTEGER";
        if (sqlite3_exec(db, query.c_str(), 0, 0, 0) != SQLITE_OK) {
            std::cerr << "Error adding info column: " << sqlite3_errmsg(db) << std::endl;
        }
    }
}

// Function to convert FEN string to a bitboard for a specific piece
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

// Function to convert en passant square to bitboard
uint64_t EnPassantToBitboard(const std::string& FEN) {
    std::map<char, int> fileMap = {{'a', 7}, {'b', 6}, {'c', 5}, {'d', 4}, {'e', 3}, {'f', 2}, {'g', 1}, {'h', 0}};
    size_t spaceCount = 0;
    size_t enPassantPos = 0;

    // Locate the en passant field (5th field in the FEN string)
    for (size_t i = 0; i < FEN.size(); ++i) {
        if (FEN[i] == ' ') {
            ++spaceCount;
            if (spaceCount == 3) {
                enPassantPos = i + 1;
                break;
            }
        }
    }

    // If there's no en passant square or it's not valid, return 0
    if (enPassantPos == 0 || FEN.substr(enPassantPos, 1) == "-") {
        return 0;
    }

    char file = FEN[enPassantPos];
    char rank = FEN[enPassantPos + 1];
    
    int fileIndex = fileMap[file];
    int rankIndex = rank - '1';

    int squareIndex = rankIndex * 8 + fileIndex;
    
    return 1ull << squareIndex;
}

// Function to extract the castling rights and turn info from the FEN string
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

// Function to update the 'info' column
// void updateInfoColumn(sqlite3* db, const std::string& tableName) {
//     sqlite3_stmt* stmt;
//     std::string select_query = "SELECT fen, eval FROM " + tableName;
//     std::string update_query = "UPDATE " + tableName + " SET info = ? WHERE fen = ? AND eval = ?";

//     if (sqlite3_prepare_v2(db, select_query.c_str(), -1, &stmt, 0) != SQLITE_OK) {
//         std::cerr << "Error preparing select statement: " << sqlite3_errmsg(db) << std::endl;
//         return;
//     }

//     sqlite3_stmt* update_stmt;
//     if (sqlite3_prepare_v2(db, update_query.c_str(), -1, &update_stmt, 0) != SQLITE_OK) {
//         std::cerr << "Error preparing update statement: " << sqlite3_errmsg(db) << std::endl;
//         sqlite3_finalize(stmt);
//         return;
//     }

//     while (sqlite3_step(stmt) == SQLITE_ROW) {
//         std::string fen = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
//         std::string evaluation = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));

//         uint64_t info = extractInfoFromFEN(fen);
//         sqlite3_bind_int64(update_stmt, 1, info);
//         sqlite3_bind_text(update_stmt, 2, fen.c_str(), -1, SQLITE_STATIC);
//         sqlite3_bind_text(update_stmt, 3, evaluation.c_str(), -1, SQLITE_STATIC);

//         if (sqlite3_step(update_stmt) != SQLITE_DONE) {
//             std::cerr << "Error updating row with FEN " << fen << ": " << sqlite3_errmsg(db) << std::endl;
//         }

//         sqlite3_reset(update_stmt);
//     }

//     sqlite3_finalize(stmt);
//     sqlite3_finalize(update_stmt);
// }

// Function to update bitboards and en passant column
void updateBitboards(sqlite3* db, const std::string& tableName) {
    sqlite3_stmt* stmt;
    std::string select_query = "SELECT fen FROM " + tableName;
    std::string update_query = "UPDATE " + tableName + " SET w_P_bitboard = ?, w_N_bitboard = ?, w_B_bitboard = ?, w_R_bitboard = ?, w_Q_bitboard = ?, w_K_bitboard = ?, b_p_bitboard = ?, b_n_bitboard = ?, b_b_bitboard = ?, b_r_bitboard = ?, b_q_bitboard = ?, b_k_bitboard = ?, en_passant_bitboard = ?, info = ? WHERE fen = ?";

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

    sqlite3_exec(db, "BEGIN TRANSACTION", NULL, NULL, NULL);

    // Counter to process records in batches
    const int batchSize = 500;
    int counter = 0;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        std::string fen = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        // std::string evaluation = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));

        uint64_t wP = FenToBmp(fen, 'P');
        uint64_t wN = FenToBmp(fen, 'N');
        uint64_t wB = FenToBmp(fen, 'B');
        uint64_t wR = FenToBmp(fen, 'R');
        uint64_t wQ = FenToBmp(fen, 'Q');
        uint64_t wK = FenToBmp(fen, 'K');
        uint64_t bP = FenToBmp(fen, 'p');
        uint64_t bN = FenToBmp(fen, 'n');
        uint64_t bB = FenToBmp(fen, 'b');
        uint64_t bR = FenToBmp(fen, 'r');
        uint64_t bQ = FenToBmp(fen, 'q');
        uint64_t bK = FenToBmp(fen, 'k');
        uint64_t enPassantBitboard = EnPassantToBitboard(fen);
        uint64_t info = extractInfoFromFEN(fen);

        sqlite3_bind_int64(update_stmt, 1, wP);
        sqlite3_bind_int64(update_stmt, 2, wN);
        sqlite3_bind_int64(update_stmt, 3, wB);
        sqlite3_bind_int64(update_stmt, 4, wR);
        sqlite3_bind_int64(update_stmt, 5, wQ);
        sqlite3_bind_int64(update_stmt, 6, wK);
        sqlite3_bind_int64(update_stmt, 7, bP);
        sqlite3_bind_int64(update_stmt, 8, bN);
        sqlite3_bind_int64(update_stmt, 9, bB);
        sqlite3_bind_int64(update_stmt, 10, bR);
        sqlite3_bind_int64(update_stmt, 11, bQ);
        sqlite3_bind_int64(update_stmt, 12, bK);
        sqlite3_bind_int64(update_stmt, 13, enPassantBitboard);
        sqlite3_bind_int64(update_stmt, 14, info);
        sqlite3_bind_text(update_stmt, 15, fen.c_str(), -1, SQLITE_STATIC);
        // sqlite3_bind_text(update_stmt, 15, evaluation.c_str(), -1, SQLITE_STATIC);

        if (sqlite3_step(update_stmt) != SQLITE_DONE) {
            std::cerr << "Error updating row with FEN " << fen << ": " << sqlite3_errmsg(db) << std::endl;
        }

        sqlite3_reset(update_stmt);

        counter++;
                // Commit after every batch of records
        if (counter % batchSize == 0) {
            std::cout << "Counter: " << counter << std::endl;
            sqlite3_exec(db, "COMMIT", NULL, NULL, NULL);
            sqlite3_exec(db, "BEGIN TRANSACTION", NULL, NULL, NULL);
        }
    }
    // Commit the last remaining records
    sqlite3_exec(db, "COMMIT", NULL, NULL, NULL);

    sqlite3_finalize(stmt);
    sqlite3_finalize(update_stmt);
}

// Main function
int main() {
    std::string db_name = "../data/chess_evals.db";
    std::string table_name = "one_program_test";

    sqlite3* db;
    if (sqlite3_open(db_name.c_str(), &db) != SQLITE_OK) {
        std::cerr << "Error opening database: " << sqlite3_errmsg(db) << std::endl;
        return 1;
    }

    // Add the required columns
    addColumns(db, table_name);
    addEnPassantColumn(db, table_name);
    addInfoColumn(db, table_name);

    // Update the bitboards, en passant, and info columns
    updateBitboards(db, table_name);
    // updateInfoColumn(db, table_name);

    sqlite3_close(db);
    return 0;
}


// g++ -o bin/one_to_add.exe one_to_add_them_all.cpp -lsqlite3
