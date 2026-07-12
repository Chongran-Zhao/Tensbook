// Coordinates asynchronous work whose result is allowed to update shared UI
// state only while it is still the newest request.

export function createLatestTaskGate() {
  let generation = 0;

  return {
    invalidate() {
      generation += 1;
    },

    async run(task) {
      const requestGeneration = ++generation;
      try {
        const value = await task();
        if (requestGeneration !== generation) return null;
        return { status: "fulfilled", value };
      } catch (error) {
        if (requestGeneration !== generation) return null;
        return { status: "rejected", error };
      }
    },
  };
}
