"""Pinned, inspectable action weights; downloads never execute repository code."""

import hashlib
import json
import urllib.request
from pathlib import Path

from .bootstrap import MICRODUCK, REPO

MANIFEST = Path(__file__).with_name("action_weights.json")
CACHE = REPO / ".cache/microduck_actions"
OFFICIAL = "pollen-robotics/microduck-policies"
SLOTS = {
    "sit_stand": "hd_sit",
    "ground_pick": "hd_pick",
    "roll": "hd_roll",
    "official_sit": (OFFICIAL, "alpha_sitstand.onnx"),
    "official_pick": (OFFICIAL, "alpha_ground_pick.onnx"),
    "official_roll": (OFFICIAL, "roulade.onnx"),
    "official_walk": (OFFICIAL, "alpha_walking.onnx"),
    "official_stand": (OFFICIAL, "alpha_stand.onnx"),
    "velstand": (OFFICIAL, "velstand.onnx"),
    "flamingo": ("RemiFabre/microduck-flamingo-cycle", "policy.onnx"),
    "happy_hop": ("joanfox/microduck-happy-hop", "policy.onnx"),
    "hop": ("witcheer/microduck-hop", "policy.onnx"),
    "run": ("HannesVonEssen/microduck-running", "policy.onnx"),
    "bow": ("fffiloni/microduck-polite-bow-b1d864", "policy.onnx"),
    "moonwalk": ("fffiloni/microduck-moonwalk-backward-55e6af", "policy.onnx"),
    "headstand": ("ZachGarner/microduck-headstand-fold", "policy.onnx"),
    "beak_throw": ("q2p/microduck-beak-throw", "beak_throw.onnx"),
    "groove": ("kramp/microduck-groove", "policy.onnx"),
    "pirouette": ("q2p/microduck-dizzy-duck", "policy.onnx"),
    "caught_dancing": ("microduckapp/microduck-caught-dancing", "policy.onnx"),
}


def records():
    return json.loads(MANIFEST.read_text())


def resolve(slot):
    key = SLOTS[slot]
    match = next(
        r for r in records() if (r.get("key") == key if isinstance(key, str) else (r.get("repo"), r["file"]) == key)
    )
    path = MICRODUCK / "deploy_hd1910/models" / match["file"] if isinstance(key, str) else CACHE / match["path"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run ./experiments/microduck_tasks/actions.sh --download.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != match["sha256"]:
        raise ValueError(f"Weight checksum differs: {path}. Restore the pinned model.")
    return path


def download():
    CACHE.mkdir(parents=True, exist_ok=True)
    for item in records():
        if item["origin"] != "public":
            continue
        path = CACHE / item["path"]
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]:
            continue
        with urllib.request.urlopen(item["url"], timeout=90) as response:
            data = response.read(item["bytes"] + 1)
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"Downloaded model checksum differs: {item['url']}")
        temporary = path.with_suffix(".part")
        temporary.write_bytes(data)
        temporary.replace(path)
