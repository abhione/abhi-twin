"""The memory budget is law (spec §2 / hard constraint #2) — validate that
docker/compose.yaml encodes it exactly and stays inside 128 GB with OS headroom."""

from pathlib import Path

import yaml

COMPOSE = Path(__file__).resolve().parent.parent / "docker" / "compose.yaml"

BUDGET_GB = {
    "llm": 50,
    "tts": 5,
    "stt": 4,
    "musetalk": 10,
    "qdrant": 4,
    "orchestrator": 6,
}
OS_HEADROOM_GB = 10


def load():
    return yaml.safe_load(COMPOSE.read_text())


def test_every_service_has_the_spec_memory_limit():
    services = load()["services"]
    for name, gb in BUDGET_GB.items():
        assert name in services, f"service {name} missing"
        assert services[name].get("mem_limit") == f"{gb}g", (
            f"{name}: mem_limit must be {gb}g per spec §2"
        )


def test_total_budget_within_128gb():
    total = sum(BUDGET_GB.values()) + OS_HEADROOM_GB
    assert total <= 128 - 30, f"budget {total} GB leaves too little KV-cache headroom"


def test_profiles():
    services = load()["services"]
    assert services["llm"]["profiles"] == ["core"]
    assert services["orchestrator"]["profiles"] == ["core"]
    assert services["qdrant"]["profiles"] == ["core"]
    assert services["tts"]["profiles"] == ["voice"]
    assert services["stt"]["profiles"] == ["voice"]
    assert services["musetalk"]["profiles"] == ["video"]


def test_gpu_services_mount_twin():
    services = load()["services"]
    for name in ("llm", "tts", "stt", "musetalk", "orchestrator"):
        assert "/twin:/twin" in services[name]["volumes"], f"{name} must mount /twin"
        assert services[name].get("gpus") == "all", f"{name} needs GPU access"


def test_port_contract():
    services = load()["services"]
    ports = {name: services[name]["ports"] for name in services if "ports" in services[name]}
    assert "8000:8000" in ports["llm"]
    assert "8001:8001" in ports["tts"] and "8002:8001" in ports["tts"]  # streaming variant
    assert "8003:8003" in ports["stt"]
    assert "8004:8004" in ports["musetalk"]
    assert "8080:8080" in ports["orchestrator"]
