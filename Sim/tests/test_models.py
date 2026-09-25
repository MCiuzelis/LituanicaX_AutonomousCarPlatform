"""Tests for the parts that do not need Isaac Sim: sensor models, camera geometry, track helpers.

    uv run --no-sync pytest tests/
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

from rc_sim.algorithms.geometry import CameraModel, GroundProjector  # noqa: E402
from rc_sim.config import DetectorCfg, EncoderCfg, ImuCfg  # noqa: E402
from rc_sim.detector_model import ObjectView, SyntheticDetector, nms  # noqa: E402
from rc_sim.encoder_model import EncoderModel  # noqa: E402
from rc_sim.imu_model import ImuModel, rpy_to_quat  # noqa: E402
from rc_sim.interface import CameraInfo, Detection  # noqa: E402
from rc_sim.track import ProgressTracker, TrackMap  # noqa: E402

CAM = CameraInfo(640, 360, 80.0, 0.21, 0.0, 0.22, 7.0)


def sphere_box(model, x, y, r=0.05, h=0.0875):
    u_, v_ = np.meshgrid(np.linspace(0, 2 * np.pi, 60), np.linspace(0, np.pi, 30))
    pts = np.c_[(r * np.cos(u_) * np.sin(v_)).ravel() + x, (r * np.sin(u_) * np.sin(v_)).ravel() + y,
                (h / 2 * (1 + np.cos(v_))).ravel()]
    u, v = model.project(pts)
    W, H = model.cam.width_px, model.cam.height_px
    return Detection(u.min() / W, v.min() / H, u.max() / W, v.max() / H, 0.9)


# ---- camera geometry ------------------------------------------------------------------------------
@pytest.mark.parametrize("projection", ["equidistant", "pinhole"])
def test_camera_round_trip(projection):
    m = CameraModel(CameraInfo(640, 360, 80.0, 0.2, 0.0, 0.2, 7.0, projection=projection))
    u, v = np.meshgrid(np.linspace(0, 640, 9), np.linspace(0, 360, 7))
    uu, vv = m.project_cam(m.ray_cam(u, v))
    assert np.abs(uu - u).max() < 1e-6 and np.abs(vv - v).max() < 1e-6


def test_equidistant_fov_matches_lens_spec():
    full = CameraInfo(640, 360, 120.0, 0, 0, 0.2, 0)                 # full-FoV sensor mode
    assert full.vfov_deg == pytest.approx(67.5, abs=0.1)              # spec: 66.7 deg


@pytest.mark.parametrize("x,y", [(0.9, 0.3), (1.5, -0.6), (2.5, 0.4), (4.0, 1.5)])
def test_size_ranging_recovers_position(x, y):
    g = GroundProjector(CAM)
    est = g.object_position(sphere_box(g.model, x, y), max_range=10, check_width=0.25)
    assert est is not None
    assert math.hypot(est[0] - x, est[1] - y) < 0.03


def test_occluded_box_is_rejected():
    g = GroundProjector(CAM)
    d = sphere_box(g.model, 2.0, 0.3)
    occluded = Detection(d.x1, d.y1 + 0.4 * d.height, d.x2, d.y2, 0.9)
    assert g.object_position(occluded, max_range=10, check_width=0.25) is None


# ---- synthetic detector ---------------------------------------------------------------------------
def make_view(ids, boxes, vis_frac=1.0):
    boxes = np.asarray(boxes, dtype=np.float64)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * 0.8
    return ObjectView(np.asarray(ids), boxes, area, area * vis_frac, boxes.copy(), np.ones(len(ids)))


def detection_rate(view, n=300, **cfg):
    c = DetectorCfg(fp_rate_per_frame=0.0, frame_drop_prob=0.0, **cfg)
    det = SyntheticDetector(c, 10, 640, 360, 458.0, 0.006, np.random.default_rng(0))
    hits = 0
    for k in range(n):
        frame, _ = det.process(k * 0.5, view)       # 0.5 s apart: independent draws
        hits += len(frame.detections)
    return hits / n


def test_big_visible_object_is_detected():
    assert detection_rate(make_view([0], [[300, 200, 340, 240]])) > 0.9


def test_tiny_object_is_not_detected():
    assert detection_rate(make_view([0], [[300, 200, 303, 203]])) == 0.0


def test_heavily_occluded_object_is_not_detected():
    assert detection_rate(make_view([0], [[300, 200, 340, 240]], vis_frac=0.15)) == 0.0


def test_nms_merges_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    assert sorted(nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5).tolist()) == [0, 2]


def test_boxes_arrive_after_latency():
    c = DetectorCfg(frame_drop_prob=0.0)
    det = SyntheticDetector(c, 1, 640, 360, 458.0, 0.006, np.random.default_rng(0))
    frame, _ = det.process(1.0, make_view([0], [[300, 200, 340, 240]]))
    assert c.latency_min_s <= frame.t_available - frame.t_capture < 0.06


# ---- IMU / encoder ----------------------------------------------------------------------------------
def test_imu_at_rest_reads_gravity():
    imu = ImuModel(ImuCfg(), np.random.default_rng(1))
    q = rpy_to_quat(0.0, 0.0, 0.3)
    imu.reset(0.0, q)
    samples = [s for k in range(1, 241) if (s := imu.step(k / 240, [0, 0, 0], [0, 0, 9.81], q)) is not None]
    assert 70 <= len(samples) <= 110
    acc = np.mean([s.accel for s in samples], axis=0)
    assert acc[2] == pytest.approx(9.81, abs=0.2)
    assert abs(np.mean([s.gyro[2] for s in samples])) < math.radians(1.0)
    assert abs(samples[-1].yaw) < math.radians(1.0)            # game rotation vector starts at 0


def test_encoder_speed_and_turns():
    cfg = EncoderCfg()
    enc = EncoderModel(cfg, np.random.default_rng(2))
    enc.reset(0.0, 0.0)
    omega = 2.0 / 0.037                                        # 2 m/s on 37 mm wheels
    for k in range(1, 241):
        enc.step(k / 240, omega * k / 240)
    expect = omega / (2 * math.pi) * cfg.motor_turns_per_wheel_turn
    r = enc.reading(1.0)
    assert r.rps == pytest.approx(expect, rel=0.02)
    assert r.turns == pytest.approx(expect, rel=0.01)


# ---- track ------------------------------------------------------------------------------------------
def test_track_spawn_point_and_progress():
    t = TrackMap()
    s, lat, _ = t.project(0.0, 0.0)
    assert abs(lat) < 0.2                                      # (0, 0) lies on the track
    p = ProgressTracker(t)
    p.reset(*t.pose_at(s)[:2])
    for k in range(1, 101):                                    # drive the centre line once round
        x, y, _ = t.pose_at(s + t.length * k / 100)
        p.update(x, y, float(k))
    assert p.laps == 1
