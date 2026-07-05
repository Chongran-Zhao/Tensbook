# Releasing Tensbook

Before tagging a release, update version-bearing files from one entry point:

```sh
scripts/prepare-release.sh <version>
```

Push an annotated tag to trigger the release workflow:

```sh
git tag -a v<version> -m "Tensbook v<version>"
git push origin v<version>
```

The workflow builds the macOS DMG and publishes the GitHub Release. To update
the Homebrew tap automatically, configure the main repository secret
`HOMEBREW_TAP_TOKEN` with write access to `Chongran-Zhao/homebrew-tensbook`.


## Stale build cache

If the project directory is renamed or moved, stale Tauri/Rust build cache may
contain old absolute paths. In that case only, run:

```sh
cargo clean
```
