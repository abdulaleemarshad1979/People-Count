"""
Robust virtual-line people counter engine.
──────────────────────────────────────────
High-accuracy directional entrance/exit and occupancy counter.

Features:
- Fast crossing detection: reliably counts fast walkers and runners in 2-3 frames.
- Deadband hysteresis: eliminates bounding-box jitter and false counts when standing on or near the line.
- True half-plane occupancy: accurately counts all persons present on the IN side of the line.
- Spatial handoff: prevents double-counting if ByteTrack drops or swaps a track ID near the line.
- Crowd throughput: supports consecutive pedestrians crossing through doorways without dropping real people.
- Rotation-invariant: works with horizontal, vertical, and diagonal counting lines.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class _Track:
    track_id: int
    point: tuple[float, float]
    box: list[float]
    box_h: float
    last_seen: float
    hits: int = 1
    side: int = 0                          # -1 = OUT, +1 = IN, 0 = in deadband / unconfirmed
    last_side_point: tuple[float, float] | None = None  # last position when on side != 0
    counted_dir: str | None = None         # "IN" or "OUT" (direction counted for this crossing)
    last_count_time: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=20))


class LineCounter:
    def __init__(
        self,
        anchor: str = "torso",
        band_ratio: float = 0.06,      # deadband half-width as fraction of box height (6%)
        min_band_px: float = 8.0,      # minimum deadband half-width in pixels
        same_track_cooldown: float = 0.8,
        handoff_window: float = 1.5,   # seconds a lost track can be associated with a new ID
        handoff_radius: float = 1.2,   # max distance in box-heights to associate lost track
        stale_timeout: float = 2.5,    # seconds before deleting inactive tracks
        span_tolerance: float = 0.25,  # 25% tolerance past ends of line segment
        clock=time.time,
    ):
        self.anchor = anchor
        self.band_ratio = band_ratio
        self.min_band_px = min_band_px
        self.same_track_cooldown = same_track_cooldown
        self.handoff_window = handoff_window
        self.handoff_radius = handoff_radius
        self.stale_timeout = stale_timeout
        self.span_tolerance = span_tolerance
        self.clock = clock

        self.count_in = 0
        self.count_out = 0
        self.tracks: dict[int, _Track] = {}
        self.events: deque = deque(maxlen=200)
        self.last_cross_time = 0.0
        self.last_cross_dir: str | None = None
        self.on_count = None  # callback(track_id, direction, timestamp)

    # ── Geometry & Anchors ───────────────────────────────────────────
    def anchor_point(self, box) -> tuple[float, float]:
        x1, y1, x2, y2 = box
        h = max(1.0, y2 - y1)
        frac = {
            "head": 0.12,
            "torso": 0.42,
            "center": 0.50,
            "feet": 0.90,
        }.get(self.anchor, 0.42)
        return ((x1 + x2) / 2.0, y1 + frac * h)

    @staticmethod
    def _line_normal(a: tuple[float, float], b: tuple[float, float], invert: bool = False) -> tuple[float, float, float]:
        """Returns (nx, ny, length) of directed line segment a -> b pointing toward IN side."""
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return (0.0, 1.0, 0.0)
        nx = -dy / length
        ny = dx / length
        if invert:
            nx, ny = -nx, -ny
        return (nx, ny, length)

    @staticmethod
    def _signed_dist(pt: tuple[float, float], a: tuple[float, float], nx: float, ny: float) -> float:
        """Positive = IN side, Negative = OUT side, Zero = on line."""
        return (pt[0] - a[0]) * nx + (pt[1] - a[1]) * ny

    def _path_hits_line(self, p1: tuple[float, float], p2: tuple[float, float],
                        a: tuple[float, float], b: tuple[float, float]) -> bool:
        """Checks whether trajectory segment p1 -> p2 intersects tripwire segment a -> b."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = a
        x4, y4 = b
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if abs(denom) < 1e-9:
            return False
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
        t = self.span_tolerance
        return 0.0 <= ua <= 1.0 and -t <= ub <= 1.0 + t

    def _is_inside(self, pt: tuple[float, float], a: tuple[float, float],
                   nx: float, ny: float) -> bool:
        """True if point is on the IN side of the line plane."""
        return self._signed_dist(pt, a, nx, ny) > 0.0

    # ── Main Tracking & Counting ─────────────────────────────────────
    def update(self, detections: list, line_px: dict, invert: bool = False) -> tuple[int, int]:
        """Updates tracks and detects virtual line crossings.

        Args:
            detections: [{"track_id": int, "box": [x1, y1, x2, y2]}, ...]
            line_px: {"x1": int, "y1": int, "x2": int, "y2": int}
            invert: swap IN and OUT directions if True

        Returns:
            (live_inside, total_active_tracks)
        """
        now = self.clock()
        a = (float(line_px["x1"]), float(line_px["y1"]))
        b = (float(line_px["x2"]), float(line_px["y2"]))
        nx, ny, length = self._line_normal(a, b, invert=invert)

        active_ids = {int(d["track_id"]) for d in detections}

        # ── Track Handoff for ID switches ────────────────────────────
        # Lost tracks that disappeared within handoff window
        lost = {
            tid: t for tid, t in self.tracks.items()
            if tid not in active_ids and (now - t.last_seen) <= self.handoff_window
        }

        # Match new IDs to nearest lost track to inherit state
        new_dets = [d for d in detections if int(d["track_id"]) not in self.tracks]
        pairs = []
        for d in new_dets:
            pt = self.anchor_point(d["box"])
            bh = max(1.0, d["box"][3] - d["box"][1])
            for tid, t in lost.items():
                dist = math.hypot(pt[0] - t.point[0], pt[1] - t.point[1])
                if dist <= self.handoff_radius * max(bh, t.box_h):
                    pairs.append((dist, int(d["track_id"]), tid))
        pairs.sort()
        used_new, used_old = set(), set()
        for _, new_id, old_id in pairs:
            if new_id in used_new or old_id in used_old:
                continue
            used_new.add(new_id)
            used_old.add(old_id)
            inherited = self.tracks.pop(old_id)
            inherited.track_id = new_id
            self.tracks[new_id] = inherited

        # ── Process Current Frame Detections ─────────────────────────
        for d in detections:
            tid = int(d["track_id"])
            box = [float(v) for v in d["box"]]
            pt = self.anchor_point(box)
            bh = max(1.0, box[3] - box[1])

            t = self.tracks.get(tid)
            if t is None:
                t = _Track(track_id=tid, point=pt, box=box, box_h=bh, last_seen=now)
                self.tracks[tid] = t
            else:
                t.hits += 1
            t.point, t.box, t.box_h, t.last_seen = pt, box, bh, now
            t.history.append(pt)

            band = max(self.min_band_px, self.band_ratio * bh)
            dist = self._signed_dist(pt, a, nx, ny)

            # Determine side: +1 = IN, -1 = OUT, 0 = deadband
            curr_side = 1 if dist > band else (-1 if dist < -band else 0)

            if curr_side == 0:
                # Inside deadband around line: do not change side state
                continue

            if t.side == 0:
                # Initial side assignment
                t.side = curr_side
                t.last_side_point = pt
                continue

            if curr_side == t.side:
                # Still on the same side: update reference point
                t.last_side_point = pt
                continue

            # ── Side Transition Detected (Crossing) ──────────────────
            origin = t.last_side_point or (t.history[0] if len(t.history) > 0 else pt)
            path_crossed = self._path_hits_line(origin, pt, a, b)

            if not path_crossed and len(t.history) >= 2:
                # Also check recent trajectory segments in case of sampled/fast movement
                for i in range(max(0, len(t.history) - 4), len(t.history) - 1):
                    if self._path_hits_line(t.history[i], pt, a, b):
                        path_crossed = True
                        break

            if path_crossed:
                direction = "IN" if curr_side == 1 else "OUT"
                can_count = (
                    t.counted_dir != direction
                    or (now - t.last_count_time) >= self.same_track_cooldown
                )

                if can_count:
                    if direction == "IN":
                        self.count_in += 1
                    else:
                        self.count_out += 1

                    t.counted_dir = direction
                    t.last_count_time = now
                    self.last_cross_time = now
                    self.last_cross_dir = direction
                    self.events.append((now, tid, direction, pt, bh))

                    if self.on_count:
                        try:
                            self.on_count(tid, direction, now)
                        except Exception:
                            pass

            # Update side to new position
            t.side = curr_side
            t.last_side_point = pt

        # ── Cleanup Stale Tracks ─────────────────────────────────────
        stale_ids = [
            tid for tid, t in self.tracks.items()
            if tid not in active_ids and (now - t.last_seen) > self.stale_timeout
        ]
        for tid in stale_ids:
            del self.tracks[tid]

        # ── Live Occupancy: People on IN side ────────────────────────
        live_inside = sum(
            1 for tid in active_ids
            if tid in self.tracks and self._is_inside(self.tracks[tid].point, a, nx, ny)
        )

        return live_inside, len(active_ids)

    def trail(self, tid: int) -> list:
        t = self.tracks.get(tid)
        return list(t.history) if t else []

    def clear_tracks(self):
        self.tracks.clear()

    def reset(self):
        self.count_in = 0
        self.count_out = 0
        self.tracks.clear()
        self.events.clear()
        self.last_cross_time = 0.0
        self.last_cross_dir = None
