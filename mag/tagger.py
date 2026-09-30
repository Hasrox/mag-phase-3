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
        if len(set(value)) != len(value):
            raise ValueError("topics must be distinct")
        unknown = [item for item in value if item not in allowed]
        if unknown:
            raise ValueError(f"topics outside vocab: {unknown}")
        return value


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
    """GBNF. An unknown enum value cannot be emitted."""
    fields = [" | ".join(f'"{item}"' for item in values(axis)) for axis in ENUM_AXES]
    topic_alts = " | ".join(f'"{item}"' for item in values("topic"))
    return f"""
root ::= "{{" ws subject "," ws action "," ws kind "," ws plural "," ws emotion "," ws intensity "," ws family "," ws setting "," ws zone "," ws topics "," ws textb "," ws profb "," ws safety "}}"
ws ::= [ \\t\\n]*
subject ::= "\\"subject\\":" ws string
action ::= "\\"action\\":" ws string
kind ::= "\\"subject_kind\\":" ws ({fields[0]})
plural ::= "\\"subject_plural\\":" ws ("true" | "false")
emotion ::= "\\"emotion\\":" ws ({fields[1]})
intensity ::= "\\"intensity\\":" ws ({fields[2]})
family ::= "\\"family\\":" ws ({fields[3]})
setting ::= "\\"setting\\":" ws ({fields[4]})
zone ::= "\\"caption_zone\\":" ws ({fields[5]})
topics ::= "\\"topics\\":" ws "[" ws topic ("," ws topic){{0,2}} ws "]"
topic ::= {topic_alts}
textb ::= "\\"text_in_image\\":" ws ("true" | "false")
profb ::= "\\"profanity_in_image\\":" ws ("true" | "false")
safety ::= "\\"safety\\":" ws ({fields[6]})
string ::= "\\"" [^"\\\\]{{1,80}} "\\""
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
    """Parse the answer. Tolerates a fence and a missing content field.

    The schema-constrained path returns bare JSON, but the chat template may wrap
    it in ```json fences, and some builds return the object under
    message.reasoning_content instead of message.content. Neither is a reason to
    throw away an otherwise valid answer.
    """
    body = _extract_json_text(text)
    return TagPayload.model_validate(json.loads(body))


def _extract_json_text(text: str) -> str:
    body = (text or "").strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1] if "\n" in body else body
        if body.rstrip().endswith("```"):
            body = body.rstrip()[: -3]
        body = body.strip()
    if not body and text:
        # The object may be embedded in a longer string; take the outer braces.
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            body = text[start:end + 1]
    return body


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
            # This server applies a schema from response_format, not a bare
            # "grammar" field. See json_schema() for why.
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "tag", "schema": json_schema()},
            },
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
