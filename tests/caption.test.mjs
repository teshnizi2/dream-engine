// The video caption must state the real re-seed interval: it is derived from the capture settings, never typed by hand.
import test from 'node:test';
import assert from 'node:assert/strict';
import { captionFor } from '../tools/caption.mjs';

test('a re-seeded video says how often, in seconds of video', () => {
  assert.equal(captionFor(30, 24), 'Dream re-seeded from the real game every **1.25 s**; in between it is pure imagination');
  assert.equal(captionFor(120, 24), 'Dream re-seeded from the real game every **5 s**; in between it is pure imagination');
  assert.equal(captionFor(144, 24), 'Dream re-seeded from the real game every **6 s**; in between it is pure imagination');
});

test('a video that is never re-seeded says so', () => {
  assert.match(captionFor(0, 24), /never re-seeded/);
  assert.match(captionFor(0, 24), /\*\*pure imagination\*\*/i);
});

test('the wake-up window of a partly re-seeded video is stated too', () => {
  assert.equal(captionFor(30, 24, 240), 'Dream re-seeded from the real game every **1.25 s** for the first 10 s, then pure imagination');
});
