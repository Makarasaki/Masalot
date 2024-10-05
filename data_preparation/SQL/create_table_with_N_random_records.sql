-- Step 1: Create the new table big_db
CREATE TABLE one_program_test AS SELECT * FROM evaluations WHERE 0;

-- Step 2: Insert 1000 random rows into big_db
INSERT INTO one_program_test
SELECT * FROM evaluations
ORDER BY RANDOM()
LIMIT 1000;