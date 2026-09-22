"""
Unit & Simulation Tests for Robust Virtual-Line People Counter (counter.py).

Tests:
  1. Stationary person with bounding box jitter on the line -> 0 IN, 0 OUT.
  2. Person hovering / pacing near the line without crossing -> 0 IN, 0 OUT.
  3. Clean single crossing (OUT -> IN) -> 1 IN, 0 OUT.
  4. Clean single crossing (IN -> OUT) -> 0 IN, 1 OUT.
  5. Fast walker / runner crossing in 2-3 frames -> 1 IN, 0 OUT.
  6. Fast runner crossing (IN -> OUT) in 2 frames -> 0 IN, 1 OUT.
  7. Tracker ID switch / handoff before line -> 1 IN (not 2).
  8. Tracker ID switch after line -> no double counting.
  9. Person walking around the line (outside segment) -> 0 counts.
 10. Consecutive pedestrians in a crowd (walking close together) -> both counted.
 11. Simultaneous walkers side-by-side -> both counted.
 12. Live Occupancy (OCC): persons anywhere in the room on IN side are accurately detected.
 13. Large batch randomized walk test (100 noisy pedestrians).
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
        c = LineCounter(clock=self.clock, band_ratio=0.06)
        # Person standing around y=240, oscillating +/- 6 pixels (within dead-band)
        for i in range(120):
            self.clock.tick()
            jitter_y = 240 + 5.0 * math.sin(i * 0.4)
            jitter_x = 320 + 3.0 * math.cos(i * 0.3)
            box = self._make_box(jitter_x, jitter_y)
            c.update([{"track_id": 1, "box": box}], self.line_px)

        self.assertEqual(c.count_in, 0, f"Expected 0 IN, got {c.count_in}")
        self.assertEqual(c.count_out, 0, f"Expected 0 OUT, got {c.count_out}")

    def test_hovering_near_line(self):
        """A person pacing on one side near the line without crossing it."""
        c = LineCounter(clock=self.clock, band_ratio=0.06)
        # Pacing between y=180 and y=220 (OUT side)
        for i in range(80):
            self.clock.tick()
            y = 200 + 15 * math.sin(i * 0.2)
            x = 250 + i * 2
            box = self._make_box(x, y)
            c.update([{"track_id": 2, "box": box}], self.line_px)

        self.assertEqual(c.count_in, 0)
        self.assertEqual(c.count_out, 0)

    def test_clean_crossing_out_to_in(self):
        """A person walks straight across from y=100 (OUT) to y=380 (IN)."""
        c = LineCounter(clock=self.clock, band_ratio=0.06)
        for y in range(100, 381, 15):
            self.clock.tick()
            box = self._make_box(300, y)
            c.update([{"track_id": 3, "box": box}], self.line_px)

        self.assertEqual(c.count_in, 1)
        self.assertEqual(c.count_out, 0)

    def test_clean_crossing_in_to_out(self):
        """A person walks straight across from y=380 (IN) to y=100 (OUT)."""
        c = LineCounter(clock=self.clock, band_ratio=0.06)
        for y in range(380, 99, -15):
            self.clock.tick()
            box = self._make_box(300, y)
            c.update([{"track_id": 4, "box": box}], self.line_px)

        self.assertEqual(c.count_in, 0)
        self.assertEqual(c.count_out, 1)

    def test_fast_walker_crossing(self):
        """A person running / walking fast across the line in only 3 frames."""
        c = LineCounter(clock=self.clock, band_ratio=0.06)
        # Frame 1: on OUT side at y=170
        self.clock.tick()
        c.update([{"track_id": 7, "box": self._make_box(300, 170)}], self.line_px)
        # Frame 2: fast jump right across line to y=270 (IN side)
        self.clock.tick()
        c.update([{"track_id": 7, "box": self._make_box(300, 270)}], self.line_px)
        # Frame 3: continues to y=370 (IN side)
        self.clock.tick()
        c.update([{"track_id": 7, "box": self._make_box(300, 370)}], self.line_px)

        self.assertEqual(c.count_in, 1, f"Fast runner must be counted IN, got {c.count_in}")
        self.assertEqual(c.count_out, 0)

    def test_fast_runner_in_to_out(self):
        """Fast runner moving from inside to outside in only 2 frames."""
        c = LineCounter(clock=self.clock, band_ratio=0.06)
        # Frame 1: on IN side at y=320
        self.clock.tick()
        c.update([{"track_id": 8, "box": self._make_box(300, 320)}], self.line_px)
        # Frame 2: jumps past line to OUT side at y=160
        self.clock.tick()
        c.update([{"track_id": 8, "box": self._make_box(300, 160)}], self.line_px)

        self.assertEqual(c.count_in, 0)
        self.assertEqual(c.count_out, 1, f"Fast runner must be counted OUT, got {c.count_out}")

    def test_id_switch_handoff_before_line(self):
        """ByteTrack drops ID 5 at the line and issues ID 99 for the same person."""
        c = LineCounter(clock=self.clock, handoff_window=1.5, handoff_radius=1.2)
        # ID 5 walks towards the line from OUT side
        for y in range(120, 220, 15):
            self.clock.tick()
            box = self._make_box(300, y)
            c.update([{"track_id": 5, "box": box}], self.line_px)

        # 1 frame gap where track is lost
        self.clock.tick()
        c.update([], self.line_px)

        # ID 99 appears at y=235 and continues to IN side at y=320
        for y in range(235, 360, 20):
            self.clock.tick()
            box = self._make_box(300, y)
            c.update([{"track_id": 99, "box": box}], self.line_px)

        self.assertEqual(c.count_in, 1, "Should count exactly 1 IN despite ID swap")
        self.assertEqual(c.count_out, 0)

    def test_id_switch_after_line_no_double_count(self):
        """Track crosses line (counted IN), ID drops on IN side, new ID appears on IN side -> no double count."""
        c = LineCounter(clock=self.clock, handoff_window=1.5, handoff_radius=1.2)
        # Track 50 crosses from OUT to IN
        for y in range(160, 300, 20):
            self.clock.tick()
            c.update([{"track_id": 50, "box": self._make_box(300, y)}], self.line_px)
        self.assertEqual(c.count_in, 1)

        # Track 50 vanishes
        self.clock.tick()
        c.update([], self.line_px)

        # Track 51 appears on IN side at y=310 and walks deeper inside to y=380
        for y in range(310, 390, 15):
            self.clock.tick()
            c.update([{"track_id": 51, "box": self._make_box(300, y)}], self.line_px)

        self.assertEqual(c.count_in, 1, "Must NOT double count when new ID appears on the inside")

    def test_walk_around_line_end(self):
        """Person crosses the y=240 threshold but past the end of the line (x=700, line ends at 540)."""
        c = LineCounter(clock=self.clock)
        for y in range(120, 361, 20):
            self.clock.tick()
            box = self._make_box(700, y)  # far beyond 540 + 25% tolerance
            c.update([{"track_id": 11, "box": box}], self.line_px)

        self.assertEqual(c.count_in, 0, "Walking around the tripwire must not trigger count")
        self.assertEqual(c.count_out, 0)

    def test_consecutive_crowd_walkers(self):
        """Two separate individuals walking through the doorway close together."""
        c = LineCounter(clock=self.clock)
        # Person A (track 101) crosses
        for y in range(140, 340, 20):
            self.clock.tick()
            c.update([{"track_id": 101, "box": self._make_box(300, y)}], self.line_px)
        self.assertEqual(c.count_in, 1)

        # Person B (track 102) crosses right behind Person A (0.5s later)
        self.clock.tick(dt=0.5)
        for y in range(140, 340, 20):
            self.clock.tick()
            c.update([{"track_id": 102, "box": self._make_box(310, y)}], self.line_px)

        self.assertEqual(c.count_in, 2, "Both consecutive individuals in doorway must be counted")

    def test_simultaneous_walkers_side_by_side(self):
        """Two people walking side-by-side at the same time must both count."""
        c = LineCounter(clock=self.clock)
        for y in range(120, 361, 15):
            self.clock.tick()
            dets = [
                {"track_id": 301, "box": self._make_box(240, y)},
                {"track_id": 302, "box": self._make_box(360, y)},
            ]
            c.update(dets, self.line_px)

        self.assertEqual(c.count_in, 2, "Both side-by-side people must be counted")
        self.assertEqual(c.count_out, 0)

    def test_live_occupancy_anywhere_in_room(self):
        """Live occupancy correctly counts persons anywhere on the IN side of the line."""
        c = LineCounter(clock=self.clock)

        # Person 1 at x=300, y=340 (directly in doorway corridor, IN side)
        # Person 2 at x=60, y=380 (standing in left corner of room, IN side)
        # Person 3 at x=300, y=140 (outside the room, OUT side)
        self.clock.tick()
        dets = [
            {"track_id": 1, "box": self._make_box(300, 340)},
            {"track_id": 2, "box": self._make_box(60, 380)},
            {"track_id": 3, "box": self._make_box(300, 140)},
        ]
        occ, active = c.update(dets, self.line_px)

        self.assertEqual(active, 3)
        self.assertEqual(occ, 2, f"Expected 2 people on IN side, got {occ}")

        # When Person 2 walks OUT
        for y in range(380, 120, -25):
            self.clock.tick()
            dets = [
                {"track_id": 1, "box": self._make_box(300, 340)},
                {"track_id": 2, "box": self._make_box(300, y)},
            ]
            occ, active = c.update(dets, self.line_px)

        self.assertEqual(c.count_out, 1, "Person 2 exiting must increment count_out")

    def test_batch_randomized_simulations(self):
        """Run 100 pedestrians through realistic noisy conditions."""
        c = LineCounter(clock=self.clock, band_ratio=0.06)
        rng = random.Random(42)

        expected_in = 0
        expected_out = 0

        tid = 100
        for p in range(100):
            direction = "IN" if (p % 2 == 0) else "OUT"
            if direction == "IN":
                y_start, y_end = 120, 360
                step = rng.randint(12, 28)
                expected_in += 1
            else:
                y_start, y_end = 360, 120
                step = -rng.randint(12, 28)
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

            self.clock.tick(dt=1.0)
            c.update([], self.line_px)

        self.assertEqual(c.count_in, expected_in, f"Expected {expected_in} IN, got {c.count_in}")
        self.assertEqual(c.count_out, expected_out, f"Expected {expected_out} OUT, got {c.count_out}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
