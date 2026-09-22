"""
Robust virtual-line people counter.

Why this exists
───────────────
The old counter fired on *any* sign of a crossing (5 anchor points, 8 frames
of history, one-frame side flips, ID "re-identification" within 250 px).
Every one of those triggers also fires on box jitter, so one person standing
near the line, or one ID switch, produced IN, OUT, IN, OUT...

This counter uses a single rule that jitter cannot satisfy:

    A person is counted only when their anchor point has been STABLY on
    side A (outside a dead-band around the line for N frames) and then
    becomes STABLY on side B, and the path between those two stable
    positions actually passes through the drawn line segment.

Plus three guards:
    * min_hits      – ghost tracks that live 1-2 frames never count
    * handoff       – if the tracker drops an ID and gives the same person a
                      new one, the new ID inherits the old one's state, so
                      the crossing is counted once, not twice
    * dup suppress  – if a *vanished* track just counted in the same place,
                      a brand-new track counting the same direction there
                      is treated as the same person (ID switch at the line)

No OpenCV / YOLO imports here, so it can be unit-tested on its own.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class _Track:
    point: tuple
    box: list
    box_h: float
    last_seen: float
    hits: int = 1
    stable_side: int | None = None      # confirmed side: +1 / -1 / None (unknown yet)
    stable_point: tuple | None = None   # last anchor position on the stable side
    pending_side: int = 0
    pending_count: int = 0
    last_count_time: float = 0.0
    last_count_dir: str | None = None
    history: deque = field(default_factory=lambda: deque(maxlen=30))


class LineCounter:
    def __init__(
        self,
        anchor: str = "torso",
        band_ratio: float = 0.12,     # dead-band half-width as fraction of person box height
        min_band_px: float = 10.0,    # ...but never thinner than this
        confirm_frames: int = 2,      # consecutive detector frames needed to confirm a side
        min_hits: int = 3,            # track must be seen this many times before it can count
        same_track_cooldown: float = 1.0,
        handoff_window: float = 1.5,  # seconds a lost track can be taken over by a new ID
        handoff_radius: float = 0.8,  # ...within this many box-heights
        dup_window: float = 1.5,
        dup_radius: float = 0.6,
        stale_timeout: float = 2.0,
        span_tolerance: float = 0.10,  # accept crossings up to 10% past each end of the line
        clock=time.time,
    ):
        self.anchor = anchor
        self.band_ratio = band_ratio
        self.min_band_px = min_band_px
        self.confirm_frames = confirm_frames
        self.min_hits = min_hits
        self.same_track_cooldown = same_track_cooldown
        self.handoff_window = handoff_window
        self.handoff_radius = handoff_radius
        self.dup_window = dup_window
        self.dup_radius = dup_radius
        self.stale_timeout = stale_timeout
        self.span_tolerance = span_tolerance
        self.clock = clock

        self.count_in = 0
        self.count_out = 0
        self.tracks: dict[int, _Track] = {}
        self.events: deque = deque(maxlen=200)   # (time, tid, dir, point, box_h)
        self.last_cross_time = 0.0
        self.last_cross_dir: str | None = None
        self.on_count = None                      # optional callback(tid, direction, t)

    # ── geometry ──────────────────────────────────────────────────────
    def anchor_point(self, box) -> tuple:
        x1, y1, x2, y2 = box
        h = y2 - y1
        frac = {"head": 0.12, "torso": 0.42, "center": 0.50, "feet": 0.92}.get(self.anchor, 0.42)
        return ((x1 + x2) / 2.0, y1 + frac * h)

    @staticmethod
    def _signed_dist(pt, a, b) -> float:
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return 0.0
        return (dx * (pt[1] - a[1]) - dy * (pt[0] - a[0])) / length

    def _path_hits_line(self, p1, p2, a, b) -> bool:
        """Does the segment p1->p2 cross the (slightly extended) line segment a->b?"""
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

    # ── main update ───────────────────────────────────────────────────
    def update(self, detections: list, line_px: dict, invert: bool = False):
        """detections: [{"track_id": int, "box": [x1,y1,x2,y2]}, ...]
        Returns (people_on_in_side_now, active_track_count)."""
        now = self.clock()
        a = (float(line_px["x1"]), float(line_px["y1"]))
        b = (float(line_px["x2"]), float(line_px["y2"]))
        in_side = -1 if invert else 1

        active_ids = {int(d["track_id"]) for d in detections}

        # Tracks that exist but were not seen this frame are candidates for handoff
        lost = {
            tid: t for tid, t in self.tracks.items()
            if tid not in active_ids
            and (now - t.last_seen) <= self.handoff_window
            and t.hits >= self.min_hits          # never let 1-frame ghosts chain together
        }

        # Match new IDs to lost tracks one-to-one, nearest first
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
            self.tracks[new_id] = self.tracks.pop(old_id)   # inherit full state

        for d in detections:
            tid = int(d["track_id"])
            box = [float(v) for v in d["box"]]
            pt = self.anchor_point(box)
            bh = max(1.0, box[3] - box[1])

            t = self.tracks.get(tid)
            if t is None:
                t = _Track(point=pt, box=box, box_h=bh, last_seen=now)
                self.tracks[tid] = t
            else:
                t.hits += 1
            t.point, t.box, t.box_h, t.last_seen = pt, box, bh, now
            t.history.append(pt)

            band = max(self.min_band_px, self.band_ratio * bh)
            dist = self._signed_dist(pt, a, b)
            raw_side = 1 if dist > band else (-1 if dist < -band else 0)

            if raw_side == 0:
                t.pending_count = 0          # inside the dead-band: nothing is decided here
                continue

            if raw_side == t.pending_side:
                t.pending_count += 1
            else:
                t.pending_side, t.pending_count = raw_side, 1

            if t.pending_count < self.confirm_frames:
                continue

            # Side is confirmed
            if t.stable_side is None:
                t.stable_side, t.stable_point = raw_side, pt
                continue

            if raw_side == t.stable_side:
                t.stable_point = pt
                continue

            # Stable side changed: a candidate crossing
            origin = t.stable_point or pt
            t.stable_side, t.stable_point = raw_side, pt

            if t.hits < self.min_hits:
                continue
            if not self._path_hits_line(origin, pt, a, b):
                continue    # walked around the end of the line, not through it
            if now - t.last_count_time < self.same_track_cooldown:
                continue

            direction = "IN" if raw_side == in_side else "OUT"
            if self._is_duplicate(tid, direction, pt, bh, now, active_ids):
                t.last_count_time, t.last_count_dir = now, direction
                continue

            if direction == "IN":
                self.count_in += 1
            else:
                self.count_out += 1
            t.last_count_time, t.last_count_dir = now, direction
            self.last_cross_time, self.last_cross_dir = now, direction
            self.events.append((now, tid, direction, pt, bh))
            if self.on_count:
                try:
                    self.on_count(tid, direction, now)
                except Exception:
                    pass

        # Drop tracks that have been gone too long
        for tid in [tid for tid, t in self.tracks.items()
                    if tid not in active_ids and now - t.last_seen > self.stale_timeout]:
            del self.tracks[tid]

        inside_now = sum(
            1 for tid in active_ids
            if tid in self.tracks and self._is_inside(self.tracks[tid], a, b, in_side)
        )
        return inside_now, len(active_ids)

    def _is_inside(self, t: _Track, a, b, in_side) -> bool:
        """Live occupancy rule: a person counts as 'inside' only if they are
        visible right now, are a real track (not a ghost), are confirmed on the
        IN side of the line, and stand within the range of the line (between
        its two ends, not off to the left/right of it)."""
        if t.hits < self.min_hits or t.stable_side != in_side:
            return False
        dx, dy = b[0] - a[0], b[1] - a[1]
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-6:
            return False
        proj = ((t.point[0] - a[0]) * dx + (t.point[1] - a[1]) * dy) / length_sq
        return -self.span_tolerance <= proj <= 1.0 + self.span_tolerance

    def _is_duplicate(self, tid, direction, pt, bh, now, active_ids) -> bool:
        """Same direction, same place, very recent, by a track that has since vanished
        -> this is the tracker re-labelling one person, not a second person."""
        for (t_ev, ev_tid, ev_dir, ev_pt, ev_bh) in reversed(self.events):
            if now - t_ev > self.dup_window:
                break
            if ev_tid == tid or ev_dir != direction or ev_tid in active_ids:
                continue
            if math.hypot(pt[0] - ev_pt[0], pt[1] - ev_pt[1]) <= self.dup_radius * max(bh, ev_bh):
                return True
        return False

    # ── housekeeping ──────────────────────────────────────────────────
    def trail(self, tid) -> list:
        t = self.tracks.get(tid)
        return list(t.history) if t else []

    def clear_tracks(self):
        """Call when the line is moved: old side information is meaningless."""
        self.tracks.clear()

    def reset(self):
        self.count_in = 0
        self.count_out = 0
        self.tracks.clear()
        self.events.clear()
        self.last_cross_time = 0.0
        self.last_cross_dir = None
