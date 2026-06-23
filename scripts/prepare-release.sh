#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/prepare-release.sh <version> [--sha256 <dmg_sha256>] [--no-cargo-check]

Examples:
  scripts/prepare-release.sh 1.1.0
  scripts/prepare-release.sh 1.1.0 --sha256 <64-hex-sha>

Updates version-bearing files for a TensorForge release:
  - Cargo.toml
  - src-tauri/Cargo.toml
  - src-tauri/tauri.conf.json
  - ui/index.html
  - packaging/tensorforge-cask.rb

When --sha256 is omitted, the cask checksum is reset to the release placeholder.
EOF
}

version=""
sha256=""
run_cargo_check=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --sha256)
      [[ $# -ge 2 ]] || { echo "error: --sha256 requires a value" >&2; exit 2; }
      sha256="$2"
      shift 2
      ;;
    --no-cargo-check)
      run_cargo_check=0
      shift
      ;;
    -*)
      echo "error: unknown option $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$version" ]]; then
        echo "error: version was already set to $version" >&2
        exit 2
      fi
      version="$1"
      shift
      ;;
  esac
done

if [[ -z "$version" ]]; then
  usage >&2
  exit 2
fi

if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.+-][0-9A-Za-z.-]+)?$ ]]; then
  echo "error: version must look like 1.1.0" >&2
  exit 2
fi

if [[ -n "$sha256" && ! "$sha256" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "error: --sha256 must be a 64-character hex digest" >&2
  exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export TF_VERSION="$version"
export TF_SHA256="${sha256:-REPLACE_WITH_AARCH64_APPLE_DARWIN_DMG_SHA256}"

python3 <<'PY'
from pathlib import Path
import os
import re

version = os.environ["TF_VERSION"]
sha256 = os.environ["TF_SHA256"]

parts = version.split("-", 1)[0].split("+", 1)[0].split(".")
display_version = ".".join(parts[:2]) if len(parts) == 3 and parts[2] == "0" else version

def replace_once(path, pattern, replacement):
    p = Path(path)
    text = p.read_text()
    new, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"expected one replacement in {path}: {pattern}")
    p.write_text(new)

replace_once("Cargo.toml", r'^version = "[^"]+"', f'version = "{version}"')
replace_once("src-tauri/Cargo.toml", r'^version = "[^"]+"', f'version = "{version}"')
replace_once("src-tauri/tauri.conf.json", r'("version": ")[^"]+(")', rf'\g<1>{version}\2')

index = Path("ui/index.html")
text = index.read_text()
text = re.sub(r'TensorForge v[0-9A-Za-z.+-]+(?:\.[0-9A-Za-z.+-]+)*', f'TensorForge v{display_version}', text)
text = re.sub(r'\u2014 v[0-9A-Za-z.+-]+(?:\.[0-9A-Za-z.+-]+)*', f'\u2014 v{display_version}', text)
index.write_text(text)

cask = Path("packaging/tensorforge-cask.rb")
text = cask.read_text()
text = re.sub(r'version "[^"]+"', f'version "{version}"', text, count=1)
text = re.sub(r'sha256 "[^"]+"', f'sha256 "{sha256}"', text, count=1)
text = re.sub(
    r'releases/download/v[^/]+/TensorForge-v#\{version\}-aarch64-apple-darwin\.dmg',
    r'releases/download/v#{version}/TensorForge-v#{version}-aarch64-apple-darwin.dmg',
    text,
    count=1,
)
cask.write_text(text)

print(f"updated TensorForge release metadata to {version} (UI v{display_version})")
PY

if [[ "$run_cargo_check" -eq 1 ]]; then
  cargo check --workspace >/dev/null
fi

echo "done"
