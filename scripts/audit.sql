-- A-1
-- No impression references a gold image or a gold sound. License notes are not a gate.
SELECT i.id
FROM impressions i
JOIN assets a ON a.id = i.image_id
WHERE a.is_gold = 1
UNION ALL
SELECT i.id
FROM impressions i
JOIN assets a ON a.id = i.sound_id
WHERE a.is_gold = 1;

-- A-2
-- No image twice inside its cooldown window. No composition repeats inside 200.
SELECT later.id
FROM impressions later
JOIN impressions earlier
  ON earlier.profile_id = later.profile_id
 AND earlier.image_id = later.image_id
 AND earlier.id < later.id
WHERE (
  SELECT COUNT(*) FROM impressions mid
  WHERE mid.profile_id = later.profile_id
    AND mid.id > earlier.id AND mid.id < later.id
) < COALESCE(earlier.cooldown_k, 1)
UNION ALL
SELECT later.id
FROM impressions later
JOIN impressions earlier
  ON earlier.profile_id = later.profile_id
 AND earlier.composition_id = later.composition_id
 AND earlier.id < later.id
WHERE later.composition_id IS NOT NULL
  AND (
    SELECT COUNT(*) FROM impressions mid
    WHERE mid.profile_id = later.profile_id
      AND mid.id > earlier.id AND mid.id < later.id
  ) < 200;

-- A-3
-- No two consecutive impressions share a template, or a sound other than silence.
WITH ordered AS (
  SELECT id, profile_id, template_id, sound_id,
         LAG(template_id) OVER (PARTITION BY profile_id ORDER BY id) AS prev_tpl,
         LAG(sound_id) OVER (PARTITION BY profile_id ORDER BY id) AS prev_snd
  FROM impressions
)
SELECT id FROM ordered
WHERE (template_id IS NOT NULL AND template_id = prev_tpl)
   OR (sound_id IS NOT NULL AND sound_id = prev_snd);

-- A-4
-- Rated rows have stars 1..5 and the fixed reward. Skipped rows have null reward.
SELECT id FROM impressions
WHERE outcome = 'rated'
  AND (
    stars NOT BETWEEN 1 AND 5
    OR reward IS NULL
    OR ABS(reward - (stars - 1) * 0.25) > 1e-9
  )
UNION ALL
SELECT id FROM impressions
WHERE outcome = 'skipped' AND reward IS NOT NULL;

-- A-5
-- Every completed block has exactly one holdout row.
SELECT profile_id, block_id
FROM impressions
WHERE block_id IS NOT NULL
GROUP BY profile_id, block_id
HAVING COUNT(*) = 5
   AND SUM(CASE WHEN policy = 'holdout' THEN 1 ELSE 0 END) != 1;

-- A-6
-- Every active image has a live value on each required axis, a topic, safety ok,
-- and a tag from an accepted run or a human.
SELECT a.id
FROM assets a
WHERE a.kind = 'image' AND a.state = 'active'
  AND (
    NOT EXISTS (
      SELECT 1 FROM tags t
      WHERE t.asset_id = a.id AND t.superseded = 0 AND t.axis = 'safety' AND t.value = 'ok'
    )
    OR NOT EXISTS (
      SELECT 1 FROM tags t
      WHERE t.asset_id = a.id AND t.superseded = 0 AND t.axis = 'topic'
    )
    OR EXISTS (
      SELECT 1 FROM (
        SELECT 'emotion' AS axis
        UNION SELECT 'intensity'
        UNION SELECT 'family'
        UNION SELECT 'caption_zone'
      ) need
      WHERE (
        SELECT COUNT(DISTINCT t.value) FROM tags t
        WHERE t.asset_id = a.id AND t.superseded = 0 AND t.axis = need.axis
      ) != 1
    )
    OR EXISTS (
      SELECT 1 FROM tags t
      WHERE t.asset_id = a.id AND t.superseded = 0
        AND t.source = 'model'
        AND NOT EXISTS (
          SELECT 1 FROM tag_runs r WHERE r.id = t.run_id AND r.accepted = 1
        )
    )
    OR EXISTS (
      SELECT 1 FROM tags t
      WHERE t.asset_id = a.id AND t.superseded = 0 AND t.source = 'import'
    )
  );

-- A-7
-- Every caption served has no unfilled slot and passes its length limits.
SELECT i.id
FROM impressions i
JOIN captions c ON c.image_id = i.image_id AND c.template_id = i.template_id AND c.variant = i.variant
JOIN templates t ON t.id = i.template_id
WHERE i.template_id IS NOT NULL
  AND (
    instr(c.text, '{') > 0
    OR instr(c.text, '}') > 0
    OR (length(c.text) - length(replace(c.text, char(10), '')) + 1) > t.max_lines
  );

-- A-8
-- Every name in every features_json exists in the feature registry.
SELECT i.id, j.key
FROM impressions i, json_each(i.features_json) j
WHERE i.features_json IS NOT NULL
  AND j.key NOT IN (SELECT name FROM feature_registry);
