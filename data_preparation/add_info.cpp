#include <iostream>
#include <string>
#include <sqlite3.h>
#include <bitset>

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

// Function to add the 'info' column if it doesn't exist
void addInfoColumn(sqlite3* db, const std::string& tableName) {
    if (!columnExists(db, tableName, "info")) {
        std::string query = "ALTER TABLE " + tableName + " ADD COLUMN info INTEGER";
        if (sqlite3_exec(db, query.c_str(), 0, 0, 0) != SQLITE_OK) {
            std::cerr << "Error adding info column: " << sqlite3_errmsg(db) << std::endl;
        }
    }
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

void updateInfoColumn(sqlite3* db, const std::string& tableName) {
    sqlite3_stmt* stmt;
    std::string select_query = "SELECT fen, eval FROM " + tableName;
    std::string update_query = "UPDATE " + tableName + " SET info = ? WHERE fen = ? AND eval = ?";

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

    // Start a transaction to speed up the batch update
    sqlite3_exec(db, "BEGIN TRANSACTION", NULL, NULL, NULL);

    // Counter to process records in batches
    const int batchSize = 1000;
    int counter = 0;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        std::string fen = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        std::string evaluation = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));

        uint64_t info = extractInfoFromFEN(fen);
        sqlite3_bind_int64(update_stmt, 1, info);
        sqlite3_bind_text(update_stmt, 2, fen.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(update_stmt, 3, evaluation.c_str(), -1, SQLITE_STATIC);

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

int main() {
    std::string tableName = "big_db_100_000";
    sqlite3* db;
    int rc = sqlite3_open("../chess_evals.db", &db);
    if (rc) {
        std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
        return 1;
    }

    // Add the 'info' column if it doesn't exist
    addInfoColumn(db, tableName);

    // Update the 'info' column for each row
    updateInfoColumn(db, tableName);

    sqlite3_close(db);
    return 0;
}
