"""Load config/mag.toml. Stdlib tomllib only."""

from __future__ import annotations

import hashlib
from pathlib import Path

import tomllib
from pydantic import BaseModel, ConfigDict

from mag.paths import CONFIG


class Meta(BaseModel):
    model_config = ConfigDict(extra="ignore")
    spec: str
    version: str


class Runtime(BaseModel):
    model_config = ConfigDict(extra="ignore")
    host: str
    port: int
    ctx_size: int
    n_gpu_layers: str
    parallel: int
    min_decode_tps: float
    min_vram_gb: float
    host_buffer_max_mib: float
    memory_delta_ratio: float
    image_long_side: int
    tag_timeout_s: int
    tag_retry_temperature: float
    tag_conf_min: float
    prompt_version: str


class Pool(BaseModel):
    model_config = ConfigDict(extra="ignore")
    min_images: int
    min_compositions: int
    image_cooldown_cap: int
    composition_cooldown: int
    skip_cooldown_mult: int
    abandon_minutes: int


class RankerCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    alpha: float
    sd_r_init: float
    block_size: int
    coverage_rated: int
    explore_window: int
    explore_cap: float
    refactor_every: int
    skip_train_value: float
    skip_train_weight: float
    lambda_bias: float
    lambda_a: float
    lambda_b: float
    lambda_c: float
    sound_top_priors: int
    sound_explore: int
    prior_strength: float


class Reward(BaseModel):
    model_config = ConfigDict(extra="ignore")
    star_1: float
    star_2: float
    star_3: float
    star_4: float
    star_5: float

    def of(self, stars: int) -> float:
        return {1: self.star_1, 2: self.star_2, 3: self.star_3, 4: self.star_4, 5: self.star_5}[stars]


class SoundCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    target_lufs: float
    max_duration_s: float
    warn_duration_s: float
    lufs_tolerance: float
    onset_db: float


class RenderCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    min_font_px: int
    max_font_px: int
    bar_height: int
    margin: int
    font_file: str


class EvalCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    bootstrap_n: int
    bootstrap_seed: int
    s2_min_n: int
    s2_min_holdout: int
    s2_min_delta: float
    sim_noise_sd: float
    null_noise_sd: float


class Config(BaseModel):
    model_config = ConfigDict(extra="ignore")
    meta: Meta
    runtime: Runtime
    pool: Pool
    ranker: RankerCfg
    reward: Reward
    sound: SoundCfg
    render: RenderCfg
    eval: EvalCfg
    path: str = ""
    config_hash: str = ""


def load_config(path: Path | None = None) -> Config:
    path = path or CONFIG
    raw = path.read_bytes()
    data = tomllib.loads(raw.decode())
    cfg = Config.model_validate(data)
    cfg.path = str(path)
    cfg.config_hash = hashlib.sha256(raw).hexdigest()
    return cfg
