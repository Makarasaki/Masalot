UPDATE test_evals_1000
SET eval = '+1000'
WHERE eval LIKE '#+%';

UPDATE test_evals_1000
SET eval = '-1000'
WHERE eval LIKE '#-%';



UPDATE test_evals_1000
SET eval = '+1000'
WHERE CAST(eval AS INTEGER) > 1000;

UPDATE test_evals_1000
SET eval = '-1000'
WHERE CAST(eval AS INTEGER) < -1000;


ALTER TABLE test_evals_1000 ADD COLUMN eval_scaled FLOAT;


UPDATE test_evals_1000
SET eval_scaled = CAST(eval AS FLOAT) / 1000;
