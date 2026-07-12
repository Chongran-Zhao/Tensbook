# Homebrew cask for the Tensbook desktop app (prebuilt DMG).
#
# Install Tensbook from the tap:
#   brew install --cask Chongran-Zhao/tensbook/tensbook
#
# This is a template. Use scripts/prepare-release.sh to bump the version. The
# release workflow fills in the real sha256 and copies the rendered cask to the
# tap repository.
cask "tensbook" do
  version "1.1.2"
  sha256 "REPLACE_WITH_AARCH64_APPLE_DARWIN_DMG_SHA256"

  url "https://github.com/Chongran-Zhao/Tensbook/releases/download/v#{version}/Tensbook-v#{version}-aarch64-apple-darwin.dmg"
  name "Tensbook"
  desc "Symbolic math notebook for tensors, calculus, and ODEs"
  homepage "https://github.com/Chongran-Zhao/Tensbook"

  app "Tensbook.app"

  postflight do
    system_command "/usr/bin/xattr",
                   args: ["-dr", "com.apple.quarantine", "#{appdir}/Tensbook.app"],
                   sudo: false
  end
end
