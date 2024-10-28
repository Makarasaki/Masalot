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

