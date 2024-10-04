Step 1: Create the new table big_db
CREATE TABLE big_db_100_000 AS SELECT * FROM evaluations WHERE 0;

-- Step 2: Insert 1000 random rows into big_db
INSERT INTO big_db_100_000
SELECT * FROM evaluations
ORDER BY RANDOM()
LIMIT 100000;