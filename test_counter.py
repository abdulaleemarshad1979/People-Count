"""
Unit & Simulation Tests for Robust Virtual-Line People Counter (counter.py).

Tests:
  1. Stationary person with bounding box jitter on the line -> 0 IN, 0 OUT.
  2. Person hovering / pacing near the line without crossing -> 0 IN, 0 OUT.
  3. Clean single crossing (OUT -> IN) -> 1 IN, 0 OUT.
  4. Clean single crossing (IN -> OUT) -> 0 IN, 1 OUT.
  5. Tracker ID switch / handoff during crossing -> 1 IN (not 2).
  6. Ghost tracks lasting 1-2 frames -> 0 counts.
  7. Person walking around the line (outside segment) -> 0 counts.
  8. Live Occupancy (OCC): only visible, confirmed people on IN side in span.
  9. Duplicate suppression for vanished tracks.
 10. Large batch randomized walk test (100 noisy pedestrians).
"""

import math
import random
import unittest
from counter import LineCounter


class FakeClock:
    def __init__(self, start=1000.0):
        self.t = start

    def tick(self, dt=0.04):  # ~25 FPS
        self.t += dt
        return self.t

    def __call__(self):
        return self.t


class TestLineCounter(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        # Horizontal line across the middle of a 640x480 frame:
        # From (100, 240) to (540, 240).
        # Standard orientation: dist > 0 is y > 240 (IN), dist < 0 is y < 240 (OUT).
        self.line_px = {"x1": 100, "y1": 240, "x2": 540, "y2": 240}

    def _make_box(self, cx, cy, w=60, h=160):
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    def test_stationary_person_with_jitter_on_line(self):
        """A person standing directly on the line with noisy box detections."""
        c = LineCounter(clock=self.clock, confirm_frames=2, band_ratio=0.12)
        # Person standing around y=240, oscillating +/- 8 pixels (well within dead-band)
        for i in range(120):
            self.clock.tick()
            jitter_y = 240 + 7.0 * math.sin(i * 0.4)
            jitter_x = 320 + 3.0 * math.cos(i * 0.3)
            box = self._make_box(jitter_x, jitter_y)
            c.update([{"track_id": 1, "box": box}], self.line_px)

        self.assertEqual(c.count_in, 0, f"Expected 0 IN, got {c.count_in}")
        self.assertEqual(c.count_out, 0, f"Expected 0 OUT, got {c.count_out}")

    def test_hovering_near_line(self):
        """A person pacing on one side near the line without crossing it."""
        c = LineCounter(clock=self.clock, confirm_frames=2, band_ratio=0.12)
        # Pacing between y=180 and y=220 (OUT side)
        for i in range(80):
            self.clock.tick()
            y = 200 + 18 * math.sin(i * 0.2)
            x = 250 + i * 2
            box = self._make_box(x, y)
            c.update([{"track_id": 2, "box": box}], self.line_px)

        self.assertEqual(c.count_in, 0)
        self.assertEqual(c.count_out, 0)

    def test_clean_crossing_out_to_in(self):
        """A person walks straight across from y=100 (OUT) to y=380 (IN)."""
        c = LineCounter(clock=self.clock, confirm_frames=2, band_ratio=0.12)
        for y in range(100, 381, 10):
            self.clock.tick()
            box = self._make_box(300, y)
            c.update([{"track_id": 3, "box": box}], self.line_px)

        self.assertEqual(c.count_in, 1)
        self.assertEqual(c.count_out, 0)

    def test_clean_crossing_in_to_out(self):
        """A person walks straight across from y=380 (IN) to y=100 (OUT)."""
        c = LineCounter(clock=self.clock, confirm_frames=2, band_ratio=0.12)
        for y in range(380, 99, -10):
            self.clock.tick()
            box = self._make_box(300, y)
            c.update([{"track_id": 4, "box": box}], self.line_px)

        self.assertEqual(c.count_in, 0)
        self.assertEqual(c.count_out, 1)

    def test_id_switch_handoff(self):
        """ByteTrack drops ID 5 at the line and issues ID 99 for the same person."""
        c = LineCounter(clock=self.clock, confirm_frames=2, handoff_window=1.5, handoff_radius=0.8)
        # ID 5 walks towards the line from OUT side
        for y in range(120, 231, 10):
            self.clock.tick()
            box = self._make_box(300, y)
            c.update([{"track_id": 5, "box": box}], self.line_px)

        # 1 frame gap where track is lost
        self.clock.tick()
        c.update([], self.line_px)

        # ID 99 appears at y=245 and continues to IN side
        for y in range(245, 361, 10):
            self.clock.tick()
            box = self._make_box(300, y)
            c.update([{"track_id": 99, "box": box}], self.line_px)

        self.assertEqual(c.count_in, 1, "Should count exactly 1 IN despite ID swap")
        self.assertEqual(c.count_out, 0)

    def test_ghost_track_ignored(self):
        """Ghost box that appears for only 1 or 2 frames must never be counted."""
        c = LineCounter(clock=self.clock, min_hits=3)
        # Frame 1: appears on OUT side
        self.clock.tick()
        c.update([{"track_id": 10, "box": self._make_box(300, 150)}], self.line_px)
        # Frame 2: jumps to IN side
        self.clock.tick()
        c.update([{"track_id": 10, "box": self._make_box(300, 320)}], self.line_px)
        # Vanishes
        self.clock.tick()
        c.update([], self.line_px)

        self.assertEqual(c.count_in, 0)
        self.assertEqual(c.count_out, 0)

    def test_walk_around_line_end(self):
        """Person crosses the y=240 threshold but past the end of the line (x=620, line ends at 540)."""
        c = LineCounter(clock=self.clock, confirm_frames=2)
        for y in range(120, 361, 10):
            self.clock.tick()
            box = self._make_box(620, y)  # > 540 + 10% tolerance (540 + 44 = 584)
            c.update([{"track_id": 11, "box": box}], self.line_px)

        self.assertEqual(c.count_in, 0, "Walking around the tripwire must not trigger count")
        self.assertEqual(c.count_out, 0)

    def test_live_occupancy(self):
        """Live occupancy tests: only visible, confirmed people on IN side within span."""
        c = LineCounter(clock=self.clock, confirm_frames=2, min_hits=3)

        # Person A (track 21) on IN side in range (x=300, y=320)
        # Person B (track 22) on OUT side (x=300, y=160)
        # Person C (track 23) on IN side but way out of range (x=700, y=320)
        for _ in range(3):
            self.clock.tick()
            dets = [
                {"track_id": 21, "box": self._make_box(300, 320)},
                {"track_id": 22, "box": self._make_box(300, 160)},
                {"track_id": 23, "box": self._make_box(700, 320)},
            ]
            occ, active = c.update(dets, self.line_px)

        self.assertEqual(active, 3)
        self.assertEqual(occ, 1, "Only Person A is confirmed on IN side within span")

        # Now Person A leaves the camera view
        self.clock.tick()
        occ, active = c.update([
            {"track_id": 22, "box": self._make_box(300, 160)},
        ], self.line_px)
        self.assertEqual(occ, 0, "Occupancy must immediately drop to 0 when person leaves view")

    def test_simultaneous_walkers_side_by_side(self):
        """Two people walking side-by-side at the same time must both count."""
        c = LineCounter(clock=self.clock, confirm_frames=2)
        # Person 301 at x=250, Person 302 at x=350 walking together from y=120 to y=360
        for y in range(120, 361, 10):
            self.clock.tick()
            dets = [
                {"track_id": 301, "box": self._make_box(250, y)},
                {"track_id": 302, "box": self._make_box(350, y)},
            ]
            c.update(dets, self.line_px)

        self.assertEqual(c.count_in, 2, "Both side-by-side people must be counted")
        self.assertEqual(c.count_out, 0)

    def test_duplicate_suppression_for_vanished_track(self):
        """If track 50 counts IN and vanishes, a brand new track counting IN at the same spot right after is suppressed as a duplicated ID."""
        c = LineCounter(clock=self.clock, confirm_frames=2, dup_window=1.5, dup_radius=0.6)
        # Track 50 crosses IN
        for y in range(120, 361, 10):
            self.clock.tick()
            c.update([{"track_id": 50, "box": self._make_box(300, y)}], self.line_px)

        self.assertEqual(c.count_in, 1)

        # Track 50 vanishes
        self.clock.tick()
        c.update([], self.line_px)

        # Track 51 appears right at the line in the same spot and crosses IN within dup_window
        for y in range(140, 361, 10):
            self.clock.tick()
            c.update([{"track_id": 51, "box": self._make_box(302, y)}], self.line_px)

        self.assertEqual(c.count_in, 1, "Duplicate count for vanished track at same spot must be suppressed")


    def test_batch_randomized_simulations(self):
        """Run 100 pedestrians through realistic noisy conditions."""
        c = LineCounter(clock=self.clock, confirm_frames=2, band_ratio=0.12)
        rng = random.Random(42)

        expected_in = 0
        expected_out = 0

        tid = 100
        for p in range(100):
            direction = "IN" if (p % 2 == 0) else "OUT"
            if direction == "IN":
                y_start, y_end = 120, 360
                step = rng.randint(8, 16)
                expected_in += 1
            else:
                y_start, y_end = 360, 120
                step = -rng.randint(8, 16)
                expected_out += 1

            x_pos = rng.randint(180, 460)
            current_tid = tid
            tid += 1

            curr_y = y_start
            while (curr_y <= y_end if step > 0 else curr_y >= y_end):
                self.clock.tick()
                # add box jitter (+/- 4px)
                noise_x = rng.uniform(-3, 3)
                noise_y = rng.uniform(-4, 4)
                box = self._make_box(x_pos + noise_x, curr_y + noise_y)
                c.update([{"track_id": current_tid, "box": box}], self.line_px)
                curr_y += step

            # Time gap between separate individuals (2.0s exceeds dup_window of 1.5s)
            self.clock.tick(dt=2.0)
            c.update([], self.line_px)

        self.assertEqual(c.count_in, expected_in, f"Expected {expected_in} IN, got {c.count_in}")
        self.assertEqual(c.count_out, expected_out, f"Expected {expected_out} OUT, got {c.count_out}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
