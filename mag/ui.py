"""One Gradio screen. Loopback only. No telemetry. No model.

Controls, in order: Generate, image, autoplay audio, five stars, Skip, Flag,
attribution chips, topic, Reset topic, Reset profile. Reset profile asks twice.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from mag.config import Config
from mag.paths import ASSETS, RENDER_CACHE
from mag.serve import FLAG_KINDS, flag, rate, reset_profile, reset_topic, score_and_pick, skip, status_line
from mag.vocab import values

UI_HOST = "127.0.0.1"
UI_PORT = 7860
LLAMA_PORT = 8080


def launch_kwargs(cache: Path | None = None, sounds: Path | None = None) -> dict:
    """Bind loopback. Serve files only from the render cache and the sound folder."""
    if UI_PORT == LLAMA_PORT:
        raise RuntimeError("UI must not bind the llama port")
    return {
        "server_name": UI_HOST,
        "server_port": UI_PORT,
        "share": False,
        "allowed_paths": [str(cache or RENDER_CACHE), str(sounds or (ASSETS / "sounds"))],
        "show_api": False,
    }


def _empty(reason: str, pool: int, candidates: int) -> tuple[None, None, str]:
    return None, None, f"empty pool: {reason}; images={pool}; compositions={candidates}"


def _sound_note(path: str) -> str:
    if not path:
        return "silence (no file; this draw has no sound)"
    file = Path(path)
    if not file.is_file():
        return f"missing audio file {file.name}"
    return f"audio {file.name}"


class PlaySession:
    """In-process session. Generate does not open a socket."""

    def __init__(self, conn, cfg: Config, archive_dir: Path, cache_dir: Path):
        self.conn = conn
        self.cfg = cfg
        self.archive_dir = archive_dir
        self.cache_dir = cache_dir
        self.topic: str | None = None
        self.impression_id: int | None = None
        self.confirm_reset = False
        self._db_lock = threading.Lock()
        from mag.serve import active_profile, open_session

        self.profile = active_profile(conn, cfg)
        self.session = open_session(conn, int(self.profile["id"]), None)

    def _status(self, extra: str = "") -> str:
        from mag.serve import rated_count

        line = status_line(
            self.topic, rated_count(self.conn, int(self.profile["id"])), self._pool_size(),
        )
        return f"{line} {extra}".strip()

    def _pool_size(self) -> int:
        from mag.serve import pool_images

        return len(pool_images(self.conn, self.topic))

    def generate(self):
        with self._db_lock:
            return self._generate()

    def _generate(self):
        self.confirm_reset = False
        result = score_and_pick(
            self.conn, self.cfg,
            topic=self.topic,
            profile_id=int(self.profile["id"]),
            session_id=int(self.session["id"]),
            cache_dir=self.cache_dir,
        )
        if result.impression_id is None:
            self.impression_id = None
            image, audio, text = _empty(result.reason, result.pool_images, result.candidate_count)
            from mag.log import get_log

            get_log().info(
                "serve empty reason=%s pool=%s candidates=%s",
                result.reason, result.pool_images, result.candidate_count,
            )
            return image, audio, self._status(text)
        self.impression_id = result.impression_id
        sound = result.sound_path or ""
        sound_note = _sound_note(sound)
        from mag.log import get_log

        get_log().info(
            "serve impression=%s policy=%s image=%s template=%s text=%r sound=%s render=%s",
            result.impression_id, result.policy, result.image_id, result.template_id,
            result.text, sound_note, result.render_path,
        )
        return result.render_path, sound or None, self._status(f"template caption | {sound_note}")

    def on_rate(self, stars: int, chips: list[str] | None):
        with self._db_lock:
            return self._on_rate(stars, chips)

    def _on_rate(self, stars: int, chips: list[str] | None):
        self.confirm_reset = False
        if self.impression_id is None:
            return self._status("nothing to rate")
        from mag.log import get_log

        get_log().info("rate impression=%s stars=%s chips=%s", self.impression_id, stars, chips or [])
        rate(self.conn, self.cfg, self.impression_id, stars, chips or [])
        self.impression_id = None
        return self._status(f"rated {stars}")

    def on_skip(self):
        with self._db_lock:
            return self._on_skip()

    def _on_skip(self):
        self.confirm_reset = False
        if self.impression_id is None:
            return self._status("nothing to skip")
        from mag.log import get_log

        get_log().info("skip impression=%s", self.impression_id)
        skip(self.conn, self.cfg, self.impression_id)
        self.impression_id = None
        return self._status("skipped")

    def on_flag(self, kind: str):
        with self._db_lock:
            return self._on_flag(kind)

    def _on_flag(self, kind: str):
        self.confirm_reset = False
        if self.impression_id is None or kind not in FLAG_KINDS:
            return self._status("nothing to flag")
        from mag.log import get_log

        get_log().info("flag impression=%s kind=%s", self.impression_id, kind)
        flag(self.conn, self.impression_id, kind)
        self.impression_id = None
        return self._status(f"flagged {kind}")

    def on_topic(self, topic: str | None):
        with self._db_lock:
            return self._on_topic(topic)

    def _on_topic(self, topic: str | None):
        self.confirm_reset = False
        self.topic = topic or None
        from mag.serve import open_session

        self.session = open_session(self.conn, int(self.profile["id"]), self.topic)
        return self._status()

    def on_reset_topic(self):
        with self._db_lock:
            return self._on_reset_topic()

    def _on_reset_topic(self):
        self.confirm_reset = False
        self.topic = None
        self.session = reset_topic(self.conn, int(self.profile["id"]))
        return self._status("topic cleared")

    def on_reset_profile(self):
        with self._db_lock:
            return self._on_reset_profile()

    def _on_reset_profile(self):
        if not self.confirm_reset:
            self.confirm_reset = True
            return self._status("click Reset profile again")
        self.profile = reset_profile(self.conn, self.cfg, self.archive_dir)
        from mag.serve import open_session

        self.session = open_session(self.conn, int(self.profile["id"]), self.topic)
        self.impression_id = None
        self.confirm_reset = False
        return self._status("profile reset")


def build_demo(play: PlaySession):
    import gradio as gr

    topics = values("topic")
    with gr.Blocks(title="MAG") as demo:
        generate = gr.Button("Generate")
        image = gr.Image(type="filepath", label="composition")
        audio = gr.Audio(autoplay=True, label="sound")
        stars = [gr.Button(str(n)) for n in range(1, 6)]
        skip_btn = gr.Button("Skip")
        flag_kind = gr.Dropdown(list(FLAG_KINDS), label="Flag")
        flag_btn = gr.Button("Flag")
        chips = gr.CheckboxGroup(["image", "caption", "sound"], label="attribution")
        topic = gr.Dropdown(topics, label="topic")
        reset_topic_btn = gr.Button("Reset topic")
        reset_profile_btn = gr.Button("Reset profile")
        status = gr.Textbox(label="status")

        generate.click(play.generate, outputs=[image, audio, status])
        for button, stars_n in zip(stars, range(1, 6)):
            button.click(
                lambda chips, n=stars_n: play.on_rate(n, chips),
                inputs=[chips], outputs=[status],
            )
        skip_btn.click(play.on_skip, outputs=[status])
        flag_btn.click(play.on_flag, inputs=[flag_kind], outputs=[status])
        topic.change(lambda value: play.on_topic(value or None), inputs=[topic], outputs=[status])
        reset_topic_btn.click(play.on_reset_topic, outputs=[status])
        reset_profile_btn.click(play.on_reset_profile, outputs=[status])
    return demo


def main() -> None:
    from mag.catalog import seed_catalog
    from mag.config import load_config
    from mag.db import connect, init_db
    from mag.inbox import absorb
    from mag.paths import DATA
    from mag.serve import abandon_stale

    cfg = load_config()
    conn = connect()
    init_db(conn)
    seed_catalog(conn)
    added = absorb(conn, cfg)
    from mag.log import get_log

    get_log().info("ui start %s", added.line())
    print(added.line())
    print("log=data/mag.log")
    for item in added.rejected:
        print(f"skip {item}")
    abandon_stale(conn, cfg.pool.abandon_minutes)
    play = PlaySession(conn, cfg, DATA / "profiles", RENDER_CACHE)
    demo = build_demo(play)
    demo.launch(**launch_kwargs())


if __name__ == "__main__":
    main()
