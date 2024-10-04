#include <iostream>
#include <string>
#include <sqlite3.h>
#include <vector>
#include <cstdint>

// Function to check if a column exists in a table
bool columnExists(sqlite3* db, const std::string& tableName, const std::string& columnName) {
    std::string query = "PRAGMA table_info(" + tableName + ");";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, 0) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            std::string existingColumnName = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            if (existingColumnName == columnName) {
                sqlite3_finalize(stmt);
                return true;
            }
        }
        sqlite3_finalize(stmt);
    }
    return false;
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

// Function to update the bitboards in the database for each FEN
void updateBitboards(sqlite3* db, const std::string& tableName) {
    sqlite3_stmt* stmt;

    // Dynamic query for selecting FEN and eval from the table
    std::string select_query = "SELECT fen, eval FROM " + tableName;
    
    // Dynamic query for updating bitboards in the table
    std::string update_query = 
        "UPDATE " + tableName + " SET "
        "w_P_bitboard = ?, w_N_bitboard = ?, w_B_bitboard = ?, w_R_bitboard = ?, w_Q_bitboard = ?, w_K_bitboard = ?, "
        "b_p_bitboard = ?, b_n_bitboard = ?, b_b_bitboard = ?, b_r_bitboard = ?, b_q_bitboard = ?, b_k_bitboard = ? "
        "WHERE fen = ? AND eval = ?";

    // Prepare the SELECT statement
    if (sqlite3_prepare_v2(db, select_query.c_str(), -1, &stmt, 0) != SQLITE_OK) {
        std::cerr << "Error preparing select statement: " << sqlite3_errmsg(db) << std::endl;
        return;
    }

    // Prepare the UPDATE statement
    sqlite3_stmt* update_stmt;
    if (sqlite3_prepare_v2(db, update_query.c_str(), -1, &update_stmt, 0) != SQLITE_OK) {
        std::cerr << "Error preparing update statement: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_finalize(stmt);
        return;
    }

    // Define the piece lists
    std::vector<std::string> whitePieces = {"P", "N", "B", "R", "Q", "K"};
    std::vector<std::string> blackPieces = {"p", "n", "b", "r", "q", "k"};

    // Iterate over the rows from the SELECT query
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        std::string fen = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        std::string evaluation = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));

        // Bind the bitboard values for white pieces
        for (size_t i = 0; i < whitePieces.size(); ++i) {
            uint64_t bitboard = FenToBmp(fen, whitePieces[i][0]);
            sqlite3_bind_int64(update_stmt, static_cast<int>(i + 1), bitboard);
        }

        // Bind the bitboard values for black pieces
        for (size_t i = 0; i < blackPieces.size(); ++i) {
            uint64_t bitboard = FenToBmp(fen, blackPieces[i][0]);
            sqlite3_bind_int64(update_stmt, static_cast<int>(i + 7), bitboard);
        }

        // Bind FEN and evaluation to the update statement
        sqlite3_bind_text(update_stmt, 13, fen.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(update_stmt, 14, evaluation.c_str(), -1, SQLITE_STATIC);

        // Execute the update query
        if (sqlite3_step(update_stmt) != SQLITE_DONE) {
            std::cerr << "Error updating row with FEN " << fen << ": " << sqlite3_errmsg(db) << std::endl;
        }

        // Reset the update statement for the next row
        sqlite3_reset(update_stmt);
    }

    // Clean up
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

    // Add columns for bitboards
    addColumns(db, tableName);

    // Update bitboards for each row
    updateBitboards(db, tableName);

    sqlite3_close(db);
    return 0;
}
