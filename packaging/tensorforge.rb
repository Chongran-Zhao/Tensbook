# Homebrew formula for the TensorForge CLI.
#
# Not yet published — this is a template for when the repository has a
# tagged release. To use locally:
#   brew install --build-from-source ./packaging/tensorforge.rb
class Tensorforge < Formula
  desc "Symbolic tensor algebra for continuum mechanics (.tens DSL)"
  homepage "https://github.com/Chongran-Zhao/TensorForge"
  url "https://github.com/Chongran-Zhao/TensorForge/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "" # fill in after tagging a release: shasum -a 256 <tarball>
  license "MIT"

  depends_on "rust" => :build

  def install
    system "cargo", "install", *std_cargo_args
  end

  test do
    (testpath/"t.tens").write <<~EOS
      F = Tensor("\\bm F", order=2, dim=3)
      C = F.T * F
      export(C, format=latex)
    EOS
    output = shell_output("#{bin}/tensorforge run #{testpath}/t.tens")
    assert_match "\\bm F^{\\mathsf{T}} \\bm F", output
  end
end
