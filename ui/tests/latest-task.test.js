import { test } from "node:test";
import assert from "node:assert/strict";
import { createLatestTaskGate } from "../latest-task.js";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

test("only the newest asynchronous task may publish a result", async () => {
  const gate = createLatestTaskGate();
  const older = deferred();
  const newer = deferred();

  const olderResult = gate.run(() => older.promise);
  const newerResult = gate.run(() => newer.promise);

  newer.resolve("new preview");
  assert.deepEqual(await newerResult, { status: "fulfilled", value: "new preview" });

  older.resolve("stale preview");
  assert.equal(await olderResult, null);
});

test("editing can invalidate an in-flight task before the debounce fires", async () => {
  const gate = createLatestTaskGate();
  const pending = deferred();
  const result = gate.run(() => pending.promise);

  gate.invalidate();
  pending.resolve("old document");

  assert.equal(await result, null);
});

test("the current rejection is returned while stale rejections are ignored", async () => {
  const gate = createLatestTaskGate();
  const stale = deferred();
  const staleResult = gate.run(() => stale.promise);
  const expected = new Error("backend unavailable");

  const currentResult = await gate.run(() => Promise.reject(expected));
  assert.equal(currentResult.status, "rejected");
  assert.equal(currentResult.error, expected);

  stale.reject(new Error("obsolete failure"));
  assert.equal(await staleResult, null);
});
