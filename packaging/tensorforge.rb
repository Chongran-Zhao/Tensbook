# Homebrew formula for the TensorForge CLI (prebuilt binaries).
#
# Install from the tap:
#   brew install Chongran-Zhao/tensorforge/tensorforge
#
# The desktop app is shipped as a cask:
#   brew install --cask Chongran-Zhao/tensorforge/tensorforge
#
# This formula ships precompiled binaries, so installing it does NOT pull in
# the Rust toolchain or LLVM (~2 GB). The binaries are built by the release
# workflow (.github/workflows/release.yml) and attached to the GitHub release;
# the sha256 values below are filled in by that workflow.
class Tensorforge < Formula
  desc "Symbolic tensor algebra for continuum mechanics (.tens DSL)"
  homepage "https://github.com/Chongran-Zhao/TensorForge"
  version "1.0.0"
  license "MIT"

  # Apple Silicon only.
  on_macos do
    on_arm do
      url "https://github.com/Chongran-Zhao/TensorForge/releases/download/v1.0.0/tensorforge-v1.0.0-aarch64-apple-darwin.tar.gz"
      sha256 "REPLACE_WITH_AARCH64_APPLE_DARWIN_SHA256"
    end
  end

  def install
    bin.install "tensorforge"
  end

  test do
    (testpath/"t.tens").write <<~EOS
      F = Tensor("\\\\bm F", order=2, dim=3)
      C = F.T * F
      export(C, format=latex)
    EOS
    output = shell_output("#{bin}/tensorforge run #{testpath}/t.tens")
    assert_match "\\\\bm F^{\\\\mathsf{T}} \\\\bm F", output
    assert_match version.to_s, shell_output("#{bin}/tensorforge --version")
  end
end
