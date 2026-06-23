import * as esbuild from "esbuild-wasm";
import { mkdir } from "node:fs/promises";

await mkdir("ui/vendor/codemirror", { recursive: true });

await esbuild.build({
  entryPoints: ["ui/codemirror-entry.js"],
  bundle: true,
  format: "esm",
  outfile: "ui/vendor/codemirror/cm.bundle.js",
  logLevel: "info",
});

await esbuild.stop();
