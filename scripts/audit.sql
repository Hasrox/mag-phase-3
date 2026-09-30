-- A-1. No impression references a gold image. License notes are not a gate.
SELECT i.id
FROM impressions i
JOIN assets a ON a.id = i.image_id
WHERE a.is_gold = 1;
