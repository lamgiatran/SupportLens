"""Write an honest availability/stream probe; never auto-install or pull a model."""

import argparse
import json
from pathlib import Path
import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.generation import make_prompt, stream_probe
from supportlens.text import JsonTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizers/qwen/tokenizer.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/ollama_probe.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))["generator"]
    report = {"status": "unavailable", "model": config["model"], "num_ctx": config["num_ctx"],
              "metric_scope": "generator_probe_not_end_to_end"}
    code = 0
    try:
        r = httpx.get(config["base_url"] + "/api/version", timeout=2)
        r.raise_for_status()
        report["ollama_version"] = r.json()
        tags = httpx.get(config["base_url"] + "/api/tags", timeout=2)
        tags.raise_for_status()
        report["model_digest"] = next((m.get("digest") for m in tags.json().get("models", [])
                                       if m.get("name") == config["model"]), None)
        tokenizer = JsonTokenizer(args.tokenizer)
        prompt = make_prompt("When should a demo request be handed to a human?", [
            {"evidence_id": "fixture-1", "text": "In this synthetic demo, requests without sufficient evidence go to a human."}
        ], tokenizer, config["num_ctx"], config["num_predict"], config["evidence_budget_tokens"])
        report.update(stream_probe(config["base_url"], config["model"], prompt,
                                   num_ctx=config["num_ctx"], num_predict=config["num_predict"]))
    except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
        report.update({"error_type": type(error).__name__, "message": str(error)})
        code = 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
