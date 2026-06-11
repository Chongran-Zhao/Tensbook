# Homebrew formula for the TensorForge CLI.
#
# Install from the tap:
#   brew install Chongran-Zhao/tensorforge/tensorforge
class Tensorforge < Formula
  desc "Symbolic tensor algebra for continuum mechanics (.tens DSL)"
  homepage "https://github.com/Chongran-Zhao/TensorForge"
  url "https://github.com/Chongran-Zhao/TensorForge/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "d18ba93c323754ce6cb40b18dfd4827277bbb309e3378f388d5b632b2070fb53"
  license "MIT"

  depends_on "rust" => :build

  def install
    # The repo is a workspace; install the root (engine + CLI) package.
    system "cargo", "install", *std_cargo_args
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
