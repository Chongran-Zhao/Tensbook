import * as esbuild from "esbuild-wasm";
import { mkdir, readFile, rename, rm } from "node:fs/promises";

await mkdir("ui/vendor/codemirror", { recursive: true });

const outfile = "ui/vendor/codemirror/cm.bundle.js";
const tmpfile = `${outfile}.tmp`;

await esbuild.build({
  entryPoints: ["ui/codemirror-entry.js"],
  bundle: true,
  format: "esm",
  outfile: tmpfile,
  logLevel: "info",
});

await esbuild.stop();

const [nextBundle, currentBundle] = await Promise.all([
  readFile(tmpfile),
  readFile(outfile).catch((error) => {
    if (error.code === "ENOENT") {
      return null;
    }
    throw error;
  }),
]);

if (currentBundle && Buffer.compare(nextBundle, currentBundle) === 0) {
  await rm(tmpfile);
} else {
  await rename(tmpfile, outfile);
}
