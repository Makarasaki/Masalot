WITH PositiveEvaluations AS (
    SELECT eval_scaled
    FROM evaluations
    WHERE eval_scaled > 0
    ORDER BY eval_scaled
),
CountPositive AS (
    SELECT COUNT(*) AS cnt
    FROM PositiveEvaluations
)
SELECT AVG(eval_scaled) AS median
FROM (
    SELECT eval_scaled
    FROM PositiveEvaluations
    LIMIT 2 - (SELECT cnt % 2 FROM CountPositive)  -- Select one or two middle rows based on even/odd count
    OFFSET (SELECT (cnt - 1) / 2 FROM CountPositive)
);