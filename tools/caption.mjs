// The caption under the video states how often the dream is re-seeded from the real game. It is derived from the capture settings
// (re-seed interval in game ticks, video fps, optional last tick of re-seeding), so it cannot disagree with what was rendered.
const fmt = (s) => String(+s.toFixed(2));

export function captionFor(wakeEvery, fps = 24, wakeUntil = Infinity) {
  if (!wakeEvery) return 'The dream is **never re-seeded** from the real game: every frame on the right is **pure imagination**';
  const every = `every **${fmt(wakeEvery / fps)} s**`;
  if (Number.isFinite(wakeUntil)) return `Dream re-seeded from the real game ${every} for the first ${fmt(wakeUntil / fps)} s, then pure imagination`;
  return `Dream re-seeded from the real game ${every}; in between it is pure imagination`;
}
