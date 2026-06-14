# Homebrew cask for the TensorForge desktop app (prebuilt DMG).
#
# Install from the tap:
#   brew install --cask Chongran-Zhao/tensorforge/tensorforge
#
# This is a template. The release workflow fills in the real sha256 and copies
# the rendered cask to the tap repository.
cask "tensorforge" do
  version "1.0.0"
  sha256 "REPLACE_WITH_AARCH64_APPLE_DARWIN_DMG_SHA256"

  url "https://github.com/Chongran-Zhao/TensorForge/releases/download/v1.0.0/TensorForge-v#{version}-aarch64-apple-darwin.dmg"
  name "TensorForge"
  desc "Symbolic tensor algebra app for continuum mechanics"
  homepage "https://github.com/Chongran-Zhao/TensorForge"

  app "TensorForge.app"
end
