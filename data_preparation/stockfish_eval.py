#!/usr/bin/env python3

import sqlite3
import chess
import chess.engine

STOCKFISH_PATH = "/home/maks/mgr/stockfish/src/stockfish"
DB_PATH = "../data/chess_evals.db"
CHUNK_SIZE = 10000  # Number of rows to process per batch
MATE_SCORE = 2000
DEPTH_LIMIT = 10
COLUMN_NAME = f"stock_d{DEPTH_LIMIT}"
TABLE_NAME = "training_dataset"

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1) Check if 'stock_d1' column exists. If not, create it.
    cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
    columns = [row[1] for row in cursor.fetchall()]  # row[1] is column name
    if COLUMN_NAME not in columns:
        print(f"Column '{COLUMN_NAME}' does not exist. Creating it...")
        cursor.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {COLUMN_NAME} REAL")
        conn.commit()

    # 2) Initialize Stockfish
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    # 3) Retrieve rows in chunks
    # cursor.execute("SELECT rowid, fen FROM training_dataset WHERE stock_d1 IS NULL")
    rows_processed = 0
    prev_rows_processed = 1

    while True:
        cursor.execute(f"SELECT rowid, fen FROM {TABLE_NAME} WHERE {COLUMN_NAME} IS NULL")
        # Fetch next batch
        rows = cursor.fetchmany(CHUNK_SIZE)
        if not rows:
            break  # No more data

        if rows_processed == prev_rows_processed:
            break

        prev_rows_processed = rows_processed

        for row_id, fen_string in rows:
            if not fen_string:
                continue

            # 4) Evaluate
            try:
                board = chess.Board(fen_string)
            except ValueError:
                print(f"Invalid FEN (rowid={row_id}): {fen_string}")
                continue

            info = engine.analyse(board, chess.engine.Limit(depth=DEPTH_LIMIT))
            score = info["score"]
            numeric_eval = score.pov(chess.WHITE).score(mate_score=MATE_SCORE)

            # 5) Update database
            cursor.execute(
                f"UPDATE {TABLE_NAME} SET {COLUMN_NAME} = ? WHERE rowid = ?",
                (numeric_eval, row_id)
            )
            rows_processed += 1

        # Commit after each batch
        conn.commit()
        print(f"Processed {rows_processed} rows so far...")

    # 6) Clean up
    engine.quit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
