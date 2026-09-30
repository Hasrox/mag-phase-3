"""Offline image tagger. llama-server only, and only after a passing gate.

The grammar is generated from vocab. A response that fails validation is
retried once at temperature 0.2 with a new seed, then the row stays untagged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mag.config import Config
from mag.gate import assert_gate_alive
from mag.paths import ASSETS
from mag.vocab import values

# Must match --media-path in scripts/tag_run.ps1. The server resolves media
# relative to this root and rejects anything outside it.
MEDIA_ROOT = ASSETS

ENUM_AXES = (
    "subject_kind",
    "emotion",
    "intensity",
    "family",
    "setting",
    "caption_zone",
    "safety",
)
WORD_LIMIT = 8
ARTICLE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)
REFUSAL = re.compile(
    r"\b(i can'?t|i cannot|i won'?t|i will not|as an ai|i must refuse|"
    r"cannot assist|not able to|against (my|the) (policy|guidelines)|"
    r"i'?m sorry,? i)\b",
    re.IGNORECASE,
)


class TagPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: str
    action: str
    subject_kind: str
    subject_plural: bool
    emotion: str
    intensity: str
    family: str
    setting: str
    caption_zone: str
    topics: list[str] = Field(min_length=1, max_length=3)
    text_in_image: bool
    profanity_in_image: bool
    safety: str

    @field_validator("subject", "action")
    @classmethod
    def _phrase(cls, value: str, info) -> str:
        text = " ".join(value.strip().lower().split())
        if info.field_name == "subject":
            text = ARTICLE.sub("", text)
        words = text.split()
        if not words or len(words) > WORD_LIMIT:
            raise ValueError(f"{info.field_name} must be 1..{WORD_LIMIT} words")
        return text

    @field_validator(
        "subject_kind", "emotion", "intensity", "family", "setting", "caption_zone", "safety",
    )
    @classmethod
    def _enum(cls, value: str, info) -> str:
        allowed = values(info.field_name)
        if value not in allowed:
            raise ValueError(f"{info.field_name} {value!r} is outside vocab")
        return value

    @field_validator("topics")
    @classmethod
    def _topics(cls, value: list[str]) -> list[str]:
        allowed = set(values("topic"))
        unknown = [item for item in value if item not in allowed]
        if unknown:
            raise ValueError(f"topics outside vocab: {unknown}")
        # The model repeats a topic rather than failing. A repeat adds no
        # information and the schema only needs 1 to 3 distinct values, so
        # collapse it instead of discarding an otherwise usable answer. Observed
        # live on 2026-09-30: ["animals", "toilet", "toilet"].
        return list(dict.fromkeys(value))


class TagAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    payload: TagPayload | None
    raw: str
    temperature: float
    seed: int
    conf: float | None
    error: str = ""


def json_schema() -> dict:
    """JSON Schema for the tag payload, built from the vocabulary tables.

    This build does not honour a bare "grammar" field on /v1/chat/completions.
    server-schema.cpp gives json_schema precedence over grammar, and the chat
    template for this model injects its own, so the GBNF below was silently
    discarded: the model answered with unquoted enums and filler whitespace.
    A JSON schema through response_format is what this server actually applies,
    and the vocabularies still make an out-of-range enum impossible.
    """
    def enum(axis: str) -> dict:
        return {"type": "string", "enum": values(axis)}

    return {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "minLength": 1, "maxLength": 120},
            "action": {"type": "string", "minLength": 1, "maxLength": 120},
            "subject_kind": enum("subject_kind"),
            "subject_plural": {"type": "boolean"},
            "emotion": enum("emotion"),
            "intensity": enum("intensity"),
            "family": enum("family"),
            "setting": enum("setting"),
            "caption_zone": enum("caption_zone"),
            "topics": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": enum("topic"),
            },
            "text_in_image": {"type": "boolean"},
            "profanity_in_image": {"type": "boolean"},
            "safety": enum("safety"),
        },
        "required": [
            "subject", "action", "subject_kind", "subject_plural", "emotion",
            "intensity", "family", "setting", "caption_zone", "topics",
            "text_in_image", "profanity_in_image", "safety",
        ],
        "additionalProperties": False,
    }


def grammar() -> str:
    """GBNF. An unknown enum value cannot be emitted.

    Two rules here are deliberate and were both found the hard way, by watching
    a real gemma-4 server on 2026-09-30:

    - ws is [ \\t]* , not [ \\t\\n]* . With \\n allowed the sampler could emit
      whitespace forever: the answer stopped dead at the first field that
      follows a ws, and filled 400 tokens with newlines and spaces.
    - topics spells out one, two and three entries instead of using {0,2}. A
      bounded repetition right after a zero-width rule is a known way to lose
      the grammar and fall back to free generation.
    """
    fields = [" | ".join(f'"{item}"' for item in values(axis)) for axis in ENUM_AXES]
    topic_alts = " | ".join(f'"{item}"' for item in values("topic"))
    return f"""
root ::= "{{" ws subject "," ws action "," ws kind "," ws plural "," ws emotion "," ws intensity "," ws family "," ws setting "," ws zone "," ws topics "," ws textb "," ws profb "," ws safety ws "}}"
ws ::= [ \\t]*
subject ::= "\\"subject\\":" ws string
action ::= "\\"action\\":" ws string
kind ::= "\\"subject_kind\\":" ws ({fields[0]})
plural ::= "\\"subject_plural\\":" ws ("true" | "false")
emotion ::= "\\"emotion\\":" ws ({fields[1]})
intensity ::= "\\"intensity\\":" ws ({fields[2]})
family ::= "\\"family\\":" ws ({fields[3]})
setting ::= "\\"setting\\":" ws ({fields[4]})
zone ::= "\\"caption_zone\\":" ws ({fields[5]})
topics ::= "\\"topics\\":" ws "[" ws topic ws "]" | "\\"topics\\":" ws "[" ws topic "," ws topic ws "]" | "\\"topics\\":" ws "[" ws topic "," ws topic "," ws topic ws "]"
topic ::= {topic_alts}
textb ::= "\\"text_in_image\\":" ws ("true" | "false")
profb ::= "\\"profanity_in_image\\":" ws ("true" | "false")
safety ::= "\\"safety\\":" ws ({fields[6]})
string ::= "\\"" [a-zA-Z 0-9]{{1,60}} "\\""
""".strip()


def prompt(rubric: str) -> str:
    return (
        "Label the image. Return only the schema. Do not write a joke, a caption, or a refusal. "
        "subject is a bare noun phrase, no article, lowercase, at most 8 words. "
        "action is a present-participle phrase, at most 8 words. "
        "Judge safety only against this rubric:\n" + rubric
    )


def is_refusal(text: str) -> bool:
    body = (text or "").strip()
    if not body:
        return True
    return REFUSAL.search(body) is not None


def parse_response(text: str) -> TagPayload:
    """Parse the answer, repairing the one defect this model has.

    Observed on a live gemma-4 server on 2026-09-30, with the grammar bound and
    verified: the grammar is honoured exactly, and the reply always looks like

        {"subject": "man in a hat", "subject_kind": animal, "topics": [animals]}

    Object keys keep their quotes; every enum value loses them. The model has no
    single token for '"animal"', so it emits the word unquoted even though the
    grammar only permits the quoted form. Re-quoting bare enum words is safe
    because the grammar has already restricted them to vocabulary values, so
    this cannot invent a label: an unknown word fails validation instead.
    """
    body = _repair_enum_quotes(_extract_json_text(text))
    return TagPayload.model_validate(json.loads(body))


# Vocabulary values may appear bare. They are re-quoted only inside a JSON
# string context, which is what an enum field always is.
_ENUM_WORDS = sorted(
    {
        value
        for axis in ENUM_AXES
        for value in values(axis)
    } | {v for v in values("topic")},
    key=len,
    reverse=True,
)
_ENUM_BARE = re.compile(
    r'(?<=[:\[,])(?P<lead>\s*)(?P<word>' + "|".join(re.escape(word) for word in _ENUM_WORDS)
    + r")(?=\s*[,\]\}])"
)


def _repair_enum_quotes(body: str) -> str:
    """Put quotes back around a bare vocabulary value in enum position.

    An already-quoted value does not match, because the lookbehind requires the
    character before the word to be a colon, comma or bracket rather than a
    quote, so this is idempotent. Whitespace after the separator is kept.
    """
    if not body:
        return body
    return _ENUM_BARE.sub(
        lambda m: f'{m.group("lead")}"{m.group("word")}"', body
    )


def _extract_json_text(text: str) -> str:
    body = (text or "").strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1] if "\n" in body else body
        if body.rstrip().endswith("```"):
            body = body.rstrip()[: -3]
        body = body.strip()
    # The gemma-4 chat template emits a literal "<|channel>thought" block into
    # content before the answer, even with --skip-chat-parsing. Drop any leading
    # channel markers, then take the outermost JSON object if the text is not
    # exactly one.
    body = _CHANNEL_PREFIX.sub("", body).strip()
    if not body.startswith("{"):
        start = body.find("{")
        end = body.rfind("}")
        if start >= 0 and end > start:
            body = body[start:end + 1]
    return body.strip()


# <|channel>thought / <|channel>final and a bare marker at the head of the reply.
_CHANNEL_PREFIX = re.compile(
    r"^\s*(?:<\|channel>[a-z_]*\s*)?(?:</?think[a-z]*>\s*)*", re.IGNORECASE
)


def _message_text(body: dict) -> str:
    """Pull the answer out of an OpenAI-shaped completion.

    Prefer message.content. Fall back to reasoning_content and to any tool-style
    field, because a thinking template can leave content empty.
    """
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    for key in ("content", "reasoning_content", "text"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _conf(body: dict) -> float | None:
    """Lowest chosen-token probability on enum fields. Null if the server omits logprobs."""
    try:
        tokens = body["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    probs: list[float] = []
    for token in tokens:
        raw = str(token.get("token", "")).strip().strip('"')
        if raw in values("topic") or any(raw in values(axis) for axis in ENUM_AXES):
            if "prob" in token:
                probs.append(float(token["prob"]))
    if not probs:
        return None
    return min(probs)


def _post(url: str, payload: dict, timeout: float, poster: Callable | None) -> dict:
    if poster is not None:
        return poster(url, payload)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


def tag_image(
    image_path: Path,
    cfg: Config,
    record_path: Path,
    model: Path,
    mmproj: Path,
    rubric: str,
    *,
    poster: Callable | None = None,
) -> list[TagAttempt]:
    """One image. Gate first. Temperature 0, then one retry at 0.2. Never a third call."""
    assert_gate_alive(record_path, model, mmproj)
    url = f"http://{cfg.runtime.host}:{cfg.runtime.port}/v1/chat/completions"
    # This server only accepts media as a path RELATIVE to --media-path, not an
    # absolute file:// URI. An absolute URI is rejected with
    # "file path is not allowed: /C:/...".
    try:
        relative = image_path.resolve().relative_to(MEDIA_ROOT.resolve()).as_posix()
        image_url = f"file://{relative}"
    except ValueError as exc:
        raise ValueError(
            f"image {image_path} is outside the media root {MEDIA_ROOT}"
        ) from exc
    attempts: list[TagAttempt] = []
    specs = ((0.0, 1), (cfg.runtime.tag_retry_temperature, 2))
    for temperature, seed in specs:
        payload = {
            "temperature": temperature,
            "seed": seed,
            # A bare "grammar" field is the shape this server honours. Probed
            # against a live gemma-4 server on 2026-09-30: with the GBNF below the
            # model emitted {"subject": "..." and could not leave the grammar.
            # response_format with a json_schema does NOT work here and is
            # silently ignored, leaving the model to answer in prose.
            "grammar": grammar(),
            # Bound the answer explicitly. Without it the server uses whatever
            # context is left, and a long vision prompt leaves none.
            "max_tokens": cfg.runtime.tag_max_tokens,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt(rubric)},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]}
            ],
        }
        try:
            body = _post(url, payload, cfg.runtime.tag_timeout_s, poster)
            text = _message_text(body)
            parsed = parse_response(text)
        except Exception as exc:  # noqa: BLE001 — invalid stays untagged after the retry
            # Keep whatever the model said. An empty raw hides truncation and
            # grammar mismatches, which is what made this take hours to find.
            raw = ""
            try:
                raw = _message_text(body)[:600]
            except Exception:
                pass
            attempts.append(TagAttempt(
                payload=None, raw=raw, temperature=temperature, seed=seed,
                conf=None, error=str(exc),
            ))
            continue
        attempts.append(TagAttempt(
            payload=parsed, raw=text, temperature=temperature, seed=seed, conf=_conf(body),
        ))
        return attempts
    return attempts

def probe_unconstrained(text: str, *, poster: Callable, url: str) -> str:
    """S-5 probe. No grammar. A refusal fails the model. Does not write tags."""
    body = poster(url, {
        "temperature": 0.0,
        "messages": [{"role": "user", "content": text}],
    })
    return str(body["choices"][0]["message"]["content"])
