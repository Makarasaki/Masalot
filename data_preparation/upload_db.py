import sqlite3
import csv

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('chess_evals.db')
cursor = conn.cursor()

# Create the table with two columns: fen and eval
cursor.execute('''
CREATE TABLE IF NOT EXISTS evaluations (
    fen TEXT PRIMARY KEY,
    eval TEXT
)
''')

conn.commit()

# Open the CSV file
with open('data/chessData.csv', 'r') as file:
    reader = csv.reader(file)
    
    # Iterate over the rows in the CSV file
    for row in reader:
        fen, eval_value = row[0], row[1]  # Adjust if the order is different
        
        # Insert the FEN and its evaluation into the table
        cursor.execute('''
        INSERT OR IGNORE INTO evaluations (fen, eval) VALUES (?, ?)
        ''', (fen, eval_value))

    conn.commit()



conn.close()
