ALTER TABLE evaluations ADD COLUMN eval_scaled FLOAT;

UPDATE evaluations
SET eval_scaled = CAST(eval AS INTEGER);

UPDATE evaluations
SET eval_scaled = 250
WHERE eval LIKE '#+%';

UPDATE evaluations
SET eval_scaled = -250
WHERE eval LIKE '#-%';



UPDATE evaluations
SET eval_scaled = 250
WHERE CAST(eval AS INTEGER) > 250;

UPDATE evaluations
SET eval_scaled = -250
WHERE CAST(eval AS INTEGER) < -250;


UPDATE evaluations
SET eval_scaled = CAST(eval_scaled AS FLOAT) / 250;

-- 


ALTER TABLE evaluations ADD COLUMN eval_not_changed FLOAT;

UPDATE evaluations
SET eval_not_changed = CAST(eval AS FLOAT);

UPDATE evaluations
SET eval_not_changed = 15319.0
WHERE eval LIKE '#+%';

UPDATE evaluations
SET eval_not_changed = -15319.0
WHERE eval LIKE '#-%';
