import sqlite3
import pandas as pd
from itertools import zip_longest

database_path = "../data/chess_evals.db"

train_chunk_size = 80_000  # e.g. 80% of 200K
test_chunk_size  =  20_000  # e.g. 20% of 200K

# 1. Open ONE connection
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

# 2. Drop (if needed) the output table
cursor.execute("DROP TABLE IF EXISTS merged_shuffled_dataset")
conn.commit()

# 3. Create chunk iterators for training & testing using same connection
train_iter = pd.read_sql(
    "SELECT * FROM training_dataset",
    con=conn,
    chunksize=train_chunk_size
)
test_iter = pd.read_sql(
    "SELECT * FROM testing_dataset",
    con=conn,
    chunksize=test_chunk_size
)

# 4. Zip over both iterators, read and write in chunks
for train_chunk, test_chunk in zip_longest(train_iter, test_iter):
    # If both are None, no more rows in either table
    if train_chunk is None and test_chunk is None:
        break

    # Combine the chunks that exist
    chunks_to_merge = []
    if train_chunk is not None:
        chunks_to_merge.append(train_chunk)
    if test_chunk is not None:
        chunks_to_merge.append(test_chunk)

    merged_chunk = pd.concat(chunks_to_merge, ignore_index=True)
    merged_chunk = merged_chunk.sample(frac=1, random_state=42).reset_index(drop=True)

    # Append to the final table in the same connection
    merged_chunk.to_sql(
        "merged_shuffled_dataset",
        conn,
        if_exists="append",
        index=False
    )

conn.close()

# 12_954_835
# 10_363_867
# 2_590_968
# 12_954_835