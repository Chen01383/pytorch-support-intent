"""Download a pinned public encoder; validate the safetensors LFS checksum."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent


def prepare_model():
    sources = json.loads((ROOT / "SOURCES.json").read_text(encoding="utf-8"))
    model_id, revision = sources["encoder_id"], sources["encoder_revision"]
    destination = ROOT / "models/minilm"
    destination.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(f"https://huggingface.co/api/models/{model_id}/revision/{revision}?blobs=true",
                                     headers={"User-Agent": "support-intent-portfolio"})
    with urllib.request.urlopen(request, timeout=90) as response:
        siblings = {row["rfilename"]: row for row in json.load(response)["siblings"]}
    filenames = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "vocab.txt", "README.md"]
    manifest = {}
    for filename in filenames:
        path = destination / filename
        url = f"https://huggingface.co/{model_id}/resolve/{revision}/{filename}"
        expected = siblings[filename].get("lfs", {}).get("sha256")
        if not path.exists():
            for attempt in range(3):
                try:
                    partial = path.with_suffix(path.suffix + ".part")
                    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
                        while block := response.read(1024 * 1024):
                            out.write(block)
                    partial.replace(path)
                    break
                except (OSError, urllib.error.URLError):
                    if attempt == 2:
                        raise
                    time.sleep(1)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected and actual != expected:
            raise ValueError(f"Encoder checksum mismatch: {filename}")
        manifest[filename] = {"sha256": actual, "bytes": path.stat().st_size, "upstream_lfs_sha256": expected}
        print(f"Verified {filename}: {path.stat().st_size:,} bytes", flush=True)
    (ROOT / "models/download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    prepare_model()
