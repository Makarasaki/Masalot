-- min value -15312.0
-- max value 15319.0
-- avg positive 567.722697113935
-- avg negative -603.244704462456
-- median positive 136.0
-- median negative -134.0
-- tanh(135/245.8) ~= 0.5

ALTER TABLE evaluations ADD COLUMN eval_scaled FLOAT;

UPDATE evaluations
SET eval_scaled = CAST(eval AS FLOAT)


UPDATE evaluations
SET eval_scaled = 15319
WHERE eval LIKE '#+%';

UPDATE evaluations
SET eval_scaled = -15319
WHERE eval LIKE '#-%';


UPDATE evaluations
SET eval_scaled = (exp(2 * (eval_scaled / 245.8)) - 1) / (exp(2 * (eval_scaled / 245.8)) + 1);







-- version2
-- tanh(135/500) ~= 0.26

ALTER TABLE evaluations ADD COLUMN eval_scaled FLOAT;

UPDATE training_dataset
SET eval_scaled = CAST(eval AS FLOAT)


UPDATE training_dataset
SET eval_scaled = 15319
WHERE eval LIKE '#+%';

UPDATE training_dataset
SET eval_scaled = -15319
WHERE eval LIKE '#-%';


UPDATE training_dataset
SET eval_scaled = (exp(2 * (eval_scaled / 500)) - 1) / (exp(2 * (eval_scaled / 500)) + 1);



-- -------------------------------------------------------------------
-- stock d1
-- find mean and average:
WITH Ordered AS (
    SELECT stock_d1
    FROM training_dataset
    WHERE stock_d1 > 0
    ORDER BY stock_d1
),
RowIndexed AS (
    SELECT stock_d1, ROW_NUMBER() OVER () AS row_num,
           COUNT(*) OVER () AS total_rows
    FROM Ordered
)
SELECT 
    AVG(stock_d1) AS mean_value,
    COUNT(stock_d1) AS count_positive_values,
    CASE 
        WHEN total_rows % 2 = 1 THEN 
            (SELECT stock_d1 FROM RowIndexed WHERE row_num = (total_rows / 2) + 1)
        ELSE 
            (SELECT AVG(stock_d1) FROM RowIndexed WHERE row_num IN ((total_rows / 2), (total_rows / 2) + 1))
    END AS median_value
FROM RowIndexed
LIMIT 1;

-- avg = 142.213078448654
-- median = 87.0
-- tanh(87/158) ~= 0.5

ALTER TABLE training_dataset ADD COLUMN stock_d1_tanh FLOAT;

UPDATE training_dataset
SET stock_d1_tanh = (exp(2 * (stock_d1 / 158)) - 1) / (exp(2 * (stock_d1 / 158)) + 1);

