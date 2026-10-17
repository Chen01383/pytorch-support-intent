"""Download immutable official BANKING77 sources, with checksum verification."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent


def download_data():
    sources = json.loads((ROOT / "SOURCES.json").read_text(encoding="utf-8"))
    base = f"https://raw.githubusercontent.com/{sources['dataset_repository']}/{sources['dataset_revision']}/"
    destination = ROOT / "data/raw"
    destination.mkdir(parents=True, exist_ok=True)
    files = {"train.csv": "banking_data/train.csv", "test.csv": "banking_data/test.csv",
             "categories.json": "banking_data/categories.json", "DATA_LICENSE.txt": "LICENSE"}
    manifest = {}
    expected_path = ROOT / "data_checksums.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8")) if expected_path.exists() else {}
    for filename, remote in files.items():
        path = destination / filename
        url = base + remote
        if not path.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "support-intent-portfolio"})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            path.write_bytes(payload)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if filename in expected and digest != expected[filename]["sha256"]:
            raise ValueError(f"Checksum mismatch: {filename}. Remove the local file and download again.")
        manifest[filename] = {"sha256": digest, "bytes": path.stat().st_size, "url": url}
    (ROOT / "data/download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"downloaded_and_verified": list(manifest)}, indent=2))
    return manifest


if __name__ == "__main__":
    download_data()
