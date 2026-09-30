-- A-1. No impression references a gold image or an asset without a license note.
SELECT i.id
FROM impressions i
JOIN assets a ON a.id = i.image_id
WHERE a.is_gold = 1 OR length(trim(a.license_note)) = 0;

-- A-2. No image appears twice inside its logged cooldown window.
SELECT later.id
FROM impressions later
JOIN impressions earlier ON earlier.image_id = later.image_id AND earlier.id < later.id
WHERE later.id - earlier.id <= CASE WHEN earlier.outcome = 'skipped' THEN later.cooldown_k * 3 ELSE later.cooldown_k END
  AND later.cooldown_k IS NOT NULL;

-- A-2b. No composition repeats inside 200 impressions.
SELECT later.id
FROM impressions later
JOIN impressions earlier
  ON earlier.image_id = later.image_id
 AND earlier.template_id = later.template_id
 AND earlier.sound_id IS later.sound_id
 AND earlier.id < later.id
WHERE later.id - earlier.id <= 200;

-- A-3. No two consecutive impressions share a template, or a sound other than silence.
SELECT later.id
FROM impressions later
JOIN impressions earlier ON earlier.id = later.id - 1
WHERE later.template_id = earlier.template_id
   OR (later.sound_id IS NOT NULL AND later.sound_id = earlier.sound_id);

-- A-4. Rated rows have stars 1 to 5 and the fixed reward. Skipped rows have null reward.
SELECT id FROM impressions
WHERE (outcome = 'rated' AND (
        stars NOT BETWEEN 1 AND 5
        OR (stars = 1 AND reward != 0.0)
        OR (stars = 2 AND reward != 0.25)
        OR (stars = 3 AND reward != 0.5)
        OR (stars = 4 AND reward != 0.75)
        OR (stars = 5 AND reward != 1.0)))
   OR (outcome = 'skipped' AND reward IS NOT NULL);

-- A-5. Every completed block has exactly one holdout row.
SELECT profile_id, block_id, SUM(policy = 'holdout') AS holdouts
FROM impressions
WHERE block_id IS NOT NULL
GROUP BY profile_id, block_id
HAVING COUNT(*) = 5 AND holdouts != 1;

-- A-6. Every active image has a live value on each required axis, a topic, safety ok, and a human or accepted-run tag.
SELECT a.id
FROM assets a
WHERE a.kind = 'image' AND a.state = 'active' AND (
    NOT EXISTS (SELECT 1 FROM tags t WHERE t.asset_id = a.id AND t.axis = 'emotion' AND t.superseded = 0)
    OR NOT EXISTS (SELECT 1 FROM tags t WHERE t.asset_id = a.id AND t.axis = 'intensity' AND t.superseded = 0)
    OR NOT EXISTS (SELECT 1 FROM tags t WHERE t.asset_id = a.id AND t.axis = 'family' AND t.superseded = 0)
    OR NOT EXISTS (SELECT 1 FROM tags t WHERE t.asset_id = a.id AND t.axis = 'setting' AND t.superseded = 0)
    OR NOT EXISTS (SELECT 1 FROM tags t WHERE t.asset_id = a.id AND t.axis = 'caption_zone' AND t.superseded = 0)
    OR NOT EXISTS (SELECT 1 FROM tags t WHERE t.asset_id = a.id AND t.axis = 'subject_kind' AND t.superseded = 0)
    OR NOT EXISTS (SELECT 1 FROM tags t WHERE t.asset_id = a.id AND t.axis = 'safety' AND t.value = 'ok' AND t.superseded = 0)
    OR NOT EXISTS (SELECT 1 FROM tags t WHERE t.asset_id = a.id AND t.axis = 'topic' AND t.superseded = 0)
    OR NOT EXISTS (
        SELECT 1 FROM tags t
        LEFT JOIN tag_runs r ON r.id = t.run_id
        WHERE t.asset_id = a.id AND t.superseded = 0
          AND (t.source = 'human' OR (t.source = 'model' AND r.accepted = 1))
    )
);

-- A-7. Every caption served has no unfilled slot and passes its length limits.
SELECT i.id
FROM impressions i
JOIN captions c ON c.image_id = i.image_id AND c.template_id = i.template_id AND c.variant = i.variant
JOIN templates t ON t.id = i.template_id
WHERE c.text LIKE '%{%'
   OR (length(c.text) - length(replace(c.text, char(10), '')) + 1) > t.max_lines;

-- A-8. Every name in features_json exists in the feature registry.
-- Populated by scripts/sync_registry.py. A missing registry row is a violation.
SELECT i.id
FROM impressions i, json_each(i.features_json)
WHERE json_each.key NOT IN (SELECT name FROM feature_registry);
