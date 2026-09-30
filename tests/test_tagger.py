"""Tagger request shape and response parsing.

Every assertion here traces to something observed on a live gemma-4 server on
2026-09-30. The point is to stop the same three mistakes coming back:

  - sending response_format instead of a bare grammar field, which this server
    silently ignores so the model answers in prose;
  - not stripping the <|channel>thought prefix the chat template puts in content;
  - reading a truncated response with no error surfaced, so a whole batch
    fails silently.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from mag.tagger import (  # noqa: E402
    TagPayload,
    _extract_json_text,
    _message_text,
    _repair_enum_quotes,
    grammar,
    json_schema,
    parse_response,
)
from mag.vocab import values  # noqa: E402

# A real reply captured from the live server, channel prefix and all.
CHANNEL_REPLY = (
    "<|channel>thought\n"
    "*   Input: \"reply with the single word ok\"\n"
    "    *   Constraint: Reply with only one word: \"ok\".\n"
)


class GrammarTests(unittest.TestCase):
    def test_grammar_covers_every_vocab_value(self) -> None:
        text = grammar()
        self.assertIn("root ::=", text)
        for axis in ("subject_kind", "emotion", "intensity", "family",
                     "setting", "caption_zone", "safety"):
            for value in values(axis):
                self.assertIn(f'"{value}"', text, f"{axis}={value} missing")

    def test_grammar_covers_every_topic(self) -> None:
        text = grammar()
        for topic in values("topic"):
            self.assertIn(f'"{topic}"', text, f"topic {topic} missing")

    def test_grammar_braces_the_object(self) -> None:
        text = grammar()
        self.assertIn('"{"', text)
        self.assertIn('"}"', text)


class JsonSchemaTests(unittest.TestCase):
    def test_schema_matches_the_model(self) -> None:
        schema = json_schema()
        self.assertEqual(sorted(schema["required"]), sorted(TagPayload.model_fields))
        self.assertFalse(schema["additionalProperties"])

    def test_schema_enums_come_from_vocab(self) -> None:
        props = json_schema()["properties"]
        self.assertEqual(props["emotion"]["enum"], values("emotion"))
        self.assertEqual(props["safety"]["enum"], values("safety"))
        self.assertEqual(props["topics"]["items"]["enum"], values("topic"))


class ExtractTests(unittest.TestCase):
    def test_strips_the_channel_prefix(self) -> None:
        self.assertEqual(_extract_json_text(CHANNEL_REPLY + '{"subject": "cat"}'),
                         '{"subject": "cat"}')

    def test_bare_object(self) -> None:
        self.assertEqual(_extract_json_text('{"subject": "x"}'), '{"subject": "x"}')

    def test_fenced_object(self) -> None:
        self.assertEqual(
            _extract_json_text('```json\n{"subject": "x"}\n```'), '{"subject": "x"}'
        )

    def test_object_embedded_in_prose(self) -> None:
        self.assertEqual(
            _extract_json_text('thinking... {"subject": "x"} done'), '{"subject": "x"}'
        )

    def test_empty_stays_empty(self) -> None:
        self.assertEqual(_extract_json_text(""), "")
        self.assertEqual(_extract_json_text(None or ""), "")


class MessageTextTests(unittest.TestCase):
    def test_prefers_content(self) -> None:
        self.assertEqual(
            _message_text({"choices": [{"message": {"content": "a", "reasoning_content": "b"}}]}),
            "a",
        )

    def test_falls_back_to_reasoning(self) -> None:
        self.assertEqual(
            _message_text({"choices": [{"message": {"content": "", "reasoning_content": "b"}}]}),
            "b",
        )

    def test_missing_shape_is_empty_not_an_exception(self) -> None:
        self.assertEqual(_message_text({}), "")
        self.assertEqual(_message_text({"choices": []}), "")


class ParseTests(unittest.TestCase):
    VALID = {
        "subject": "man in a wig", "action": "pointing at the viewer",
        "subject_kind": "person", "subject_plural": False, "emotion": "deadpan",
        "intensity": "mid", "family": "other", "setting": "indoor",
        "caption_zone": "both", "topics": ["everyday"], "text_in_image": False,
        "profanity_in_image": False, "safety": "ok",
    }

    def test_parses_a_clean_answer(self) -> None:
        import json

        self.assertEqual(parse_response(json.dumps(self.VALID)).emotion, "deadpan")

    def test_parses_through_the_channel_prefix(self) -> None:
        import json

        parsed = parse_response(CHANNEL_REPLY + json.dumps(self.VALID))
        self.assertEqual(parsed.subject, "man in a wig")

    def test_rejects_an_out_of_vocab_enum(self) -> None:
        import json

        bad = dict(self.VALID, emotion="sarcastic")
        with self.assertRaises(Exception):
            parse_response(json.dumps(bad))

    def test_repairs_bare_enum_values(self) -> None:
        """The exact shape the live server produced on 2026-09-30.

        Keys keep their quotes, every enum value loses them. The grammar is
        bound and honoured; the model just cannot emit '"animal"' as one token.
        """
        raw = (
            '{"subject": "man in a hat", "action": "pointing at the viewer", '
            '"subject_kind": animal, "subject_plural": false, "emotion": angry, '
            '"intensity": high, "family": sign, "setting": abstract, '
            '"caption_zone": top, "topics": [animals, work], '
            '"text_in_image": true, "profanity_in_image": true, "safety": ok}'
        )
        parsed = parse_response(raw)
        self.assertEqual(parsed.subject_kind, "animal")
        self.assertEqual(parsed.emotion, "angry")
        self.assertEqual(parsed.intensity, "high")
        self.assertEqual(parsed.safety, "ok")
        self.assertEqual(parsed.topics, ["animals", "work"])

    def test_repair_is_idempotent_and_quoted_input_is_untouched(self) -> None:
        import json

        once = _repair_enum_quotes('{"emotion": angry}')
        self.assertEqual(once, '{"emotion": "angry"}')
        self.assertEqual(_repair_enum_quotes(once), once)
        self.assertEqual(_repair_enum_quotes('{"emotion": "angry"}'), '{"emotion": "angry"}')

    def test_repair_does_not_touch_free_text(self) -> None:
        """A caption word that happens to equal a vocab value stays as prose."""
        text = '{"subject": "man", "action": "waiting"}'
        self.assertEqual(_repair_enum_quotes(text), text)

    def test_repeated_topics_are_collapsed_not_rejected(self) -> None:
        """Seen live: the model emits ["animals", "toilet", "toilet"].

        A repeat adds no information, so it is collapsed. An out-of-vocab topic
        is still an error.
        """
        parsed = TagPayload.model_validate(dict(self.VALID, topics=["animals", "toilet", "toilet"]))
        self.assertEqual(parsed.topics, ["animals", "toilet"])
        with self.assertRaises(Exception):
            TagPayload.model_validate(dict(self.VALID, topics=["animals", "atlantis"]))

    def test_duplicate_topics_survive_a_live_shaped_reply(self) -> None:
        raw = (
            '{"subject": "spongebob", "action": "smiling", '
            '"subject_kind": animal, "subject_plural": false, "emotion": angry, '
            '"intensity": high, "family": sign, "setting": abstract, '
            '"caption_zone": none, "topics": [animals, toilet, toilet], '
            '"text_in_image": true, "profanity_in_image": false, "safety": ok}'
        )
        self.assertEqual(parse_response(raw).topics, ["animals", "toilet"])


if __name__ == "__main__":
    unittest.main()
