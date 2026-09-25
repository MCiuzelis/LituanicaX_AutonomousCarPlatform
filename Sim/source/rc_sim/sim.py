"""The simulator: Isaac Sim scene + multi-rate sensor loop around one car.

    physics      physics_hz (240)  car dynamics, IMU + encoder models fed every step, wall contact
    camera       camera_hz  (60)   RTX render -> instance segmentation -> per-object ground truth
                                   -> SyntheticDetector -> boxes delivered after their latency
    control      control_hz (60)   CarSim.step(cmd) advances one control period and returns what
                                   the algorithm would have received on the car by then

Scene: ground plane, the car, /World/Track (one prim per object from track_objects.usdc, visual
only) and /World/Walls (walls.usdc: invisible, kinematic triangle-mesh collider; contact = failure).
Import only after Isaac Sim has been launched (run.py does that).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensor, ContactSensorCfg, Imu, ImuCfg, TiledCamera, TiledCameraCfg

from . import car as car_mod
from .car import Car
from .config import EXPORT_DIR, SimCfg
from .detector_model import FrameDebug, ObjectView, SyntheticDetector
from .encoder_model import EncoderModel
from .algorithms.geometry import CameraModel, camera_rotation
from .imu_model import ImuModel, quat_to_matrix
from .interface import CameraInfo, CarInfo, Command, DetectionFrame, SensorFrame
from .track import ProgressTracker, TrackMap

CAM_PATH = f"{car_mod.BASE_LINK}/front_camera"
GRACE_S = 0.25          # contacts right after a reset are spawn transients, not crashes
SUPPORT_DIRS = 64       # silhouette support points kept per object mesh


@dataclass
class Status:
    t: float
    x: float
    y: float
    yaw: float
    speed: float            # forward, m/s (ground truth)
    yaw_rate: float
    progress: float         # m along the centre line in the driving direction
    lateral: float          # m, + = left of the centre line
    laps: int
    lap_times: list
    done: bool = False
    reason: str = ""        # "laps", "crash", "flip", "wrong_way", "stuck", "timeout"


@dataclass
class Snapshot:
    """What the viewer and logger get after each control step (ground truth included)."""
    status: Status
    rgb: np.ndarray | None = None
    frame_debug: FrameDebug | None = None
    frame: DetectionFrame | None = None     # boxes of the latest rendered frame (None = it was dropped)
    applied: Command = field(default_factory=Command)


class CarSim:
    def __init__(self, cfg: SimCfg, track: TrackMap, gui: bool = False):
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics

        self.cfg, self.track, self.gui = cfg, track, gui
        self.dt = 1.0 / cfg.physics_hz
        render = sim_utils.RenderCfg(rendering_mode=cfg.render_mode, antialiasing_mode="FXAA", enable_dlssg=False,
                                     enable_global_illumination=False, enable_reflections=False,
                                     enable_ambient_occlusion=False, enable_translucency=False)
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=self.dt, render_interval=1, device=cfg.device, render=render))
        self.device = self.sim.device
        stage = omni.usd.get_context().get_stage()

        # ---- scene ----------------------------------------------------------------------------
        files = track.raw["files"]
        for path, spawn in (
            ("/World/GroundPlane", sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=cfg.car.ground_static_friction, dynamic_friction=cfg.car.ground_dynamic_friction,
                restitution=0.05, friction_combine_mode="multiply", restitution_combine_mode="min"))),
            ("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))),
            ("/World/Track", sim_utils.UsdFileCfg(usd_path=f"{EXPORT_DIR}/{files['objects']}")),
            ("/World/Walls", sim_utils.UsdFileCfg(usd_path=f"{EXPORT_DIR}/{files['walls']}")),
        ):
            spawn.func(path, spawn)
        walls = stage.GetPrimAtPath("/World/Walls")
        for prim in Usd.PrimRange(walls):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim).GetCollisionEnabledAttr().Set(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).GetApproximationAttr().Set("none")
        rb = UsdPhysics.RigidBodyAPI.Apply(walls)
        rb.GetRigidBodyEnabledAttr().Set(True)
        rb.GetKinematicEnabledAttr().Set(True)
        UsdGeom.Imageable(walls).MakeInvisible()

        self.car = Car(cfg.car, stage)

        # ---- sensors ----------------------------------------------------------------------------
        cc = cfg.camera
        # The detector frame uses the car's lens model (equidistant fisheye by default); Isaac renders
        # pinhole images, so the render is a wider pinhole view that covers the whole detector frame.
        self.lens = CameraModel(CameraInfo(cc.width_px, cc.height_px, cc.hfov_deg, cc.x_m, cc.y_m, cc.z_m,
                                           cc.pitch_deg, cc.yaw_deg, cc.projection))
        self.render_model = _covering_pinhole(self.lens, cc.render_scale)
        rc = self.render_model.cam
        cx, cy, cz = self.car.car_to_base(cc.x_m, cc.y_m, cc.z_m)
        s = cfg.car.robot_scale                      # the camera prim sits under the scaled robot
        self.camera = TiledCamera(TiledCameraCfg(
            prim_path=CAM_PATH, update_period=0.0,
            width=rc.width_px, height=rc.height_px,
            data_types=(["rgb"] if cc.render_rgb else []) + ["instance_id_segmentation_fast"],
            colorize_instance_id_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=4.0, horizontal_aperture=2 * 4.0 * math.tan(math.radians(rc.hfov_deg) / 2),
                clipping_range=(0.02, 40.0)),
            offset=TiledCameraCfg.OffsetCfg(pos=(cx / s, cy / s, cz / s), rot=_ros_camera_quat(cc.pitch_deg, cc.yaw_deg),
                                            convention="ros"),
        ))
        # camera pose in the base_link frame (metres), for projecting object outlines
        self._cam_pos_b = np.array([cx, cy, cz])
        self._cam_rot_b = camera_rotation(cc.pitch_deg, cc.yaw_deg)
        ic = cfg.imu
        self.imu = Imu(ImuCfg(prim_path=car_mod.BASE_LINK, update_period=0.0,
                              offset=ImuCfg.OffsetCfg(pos=self.car.car_to_base(ic.x_m, ic.y_m, ic.z_m))))
        # one sensor over every link of the car, reporting only forces against the walls
        self.contact = ContactSensor(ContactSensorCfg(prim_path=f"{car_mod.ROOT_LINK}/.*_link",
                                                      filter_prim_paths_expr=["/World/Walls"], update_period=0.0))

        self.sim.reset()
        self.car.after_sim_start()
        self.robot = self.car.robot

        # ---- models ------------------------------------------------------------------------------
        self.rng = np.random.default_rng(cfg.seed)
        self.car_info = CarInfo(
            wheelbase_m=cfg.car.wheelbase_m, max_steer_rad=cfg.max_steer_rad, wheel_radius_m=cfg.car.wheel_radius_m,
            motor_turns_per_wheel_turn=cfg.encoder.motor_turns_per_wheel_turn,
            camera=self.lens.cam,
            imu_x_m=ic.x_m, imu_z_m=ic.z_m, control_hz=cfg.control_hz, max_throttle=cfg.car.max_throttle)
        self.detector = SyntheticDetector(cfg.detector, track.n_objects, cc.width_px, cc.height_px,
                                          self.car_info.camera.fx, cc.exposure_s, self.rng)
        self.imu_model = ImuModel(ic, self.rng)
        self.encoder = EncoderModel(cfg.encoder, self.rng)
        self._outline = _object_outlines(stage, track)                             # [N, SUPPORT_DIRS, 3]
        self._lut = torch.full((1,), -1, dtype=torch.long, device=self.device)
        self._lut_size = -1
        self._substeps = max(1, round(cfg.physics_hz / cfg.control_hz))
        self._cam_every = max(1, round(cfg.physics_hz / cfg.camera_hz))
        if abs(self._substeps * cfg.control_hz - cfg.physics_hz) > 1e-6 or abs(self._cam_every * cfg.camera_hz - cfg.physics_hz) > 1e-6:
            print(f"[sim] note: control/camera rates rounded to physics_hz/{self._substeps} and /{self._cam_every}")
        print(f"[sim] physics {cfg.physics_hz:.0f} Hz | camera {cfg.physics_hz / self._cam_every:.0f} Hz "
              f"{cc.width_px}x{cc.height_px} {cc.projection} {cc.hfov_deg:.0f}x{self.lens.cam.vfov_deg:.0f} deg "
              f"(rendered {rc.width_px}x{rc.height_px} pinhole {rc.hfov_deg:.0f} deg) | control {cfg.physics_hz / self._substeps:.0f} Hz"
              f" | IMU {ic.rate_hz:.0f} Hz | track {track.n_objects} objects, {track.length:.1f} m")

    # ------------------------------------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> CarInfo:
        cfg, ep = self.cfg, self.cfg.episode
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.detector.rng = self.imu_model.rng = self.encoder.rng = self.rng
        # the car-frame origin (rear axle) goes on the spawn point, facing along the centre line
        _, _, yaw = self.track.project(ep.spawn_x, ep.spawn_y)
        yaw += math.pi if ep.reverse else 0.0
        lat = self.rng.uniform(-1, 1) * ep.spawn_lateral_jitter_m
        x, y = ep.spawn_x - math.sin(yaw) * lat, ep.spawn_y + math.cos(yaw) * lat
        yaw += math.radians(self.rng.uniform(-1, 1) * ep.spawn_yaw_jitter_deg)
        r = self.robot
        root = r.data.default_root_state.clone()
        rear = self.car.geom["rear_x"]                # base_link -> rear axle, along the car's x
        root[:, 0], root[:, 1] = x - math.cos(yaw) * rear, y - math.sin(yaw) * rear
        root[:, 2] = -self.car.geom["ground_z"] + 0.004
        root[:, 3:7] = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], device=self.device)
        root[:, 7:] = 0.0
        r.write_root_pose_to_sim(root[:, :7])
        r.write_root_velocity_to_sim(root[:, 7:])
        r.write_joint_state_to_sim(r.data.default_joint_pos.clone(), torch.zeros_like(r.data.default_joint_vel))
        r.reset()
        for sensor in (self.imu, self.camera, self.contact):
            sensor.reset()
        self.car.apply(0.0, 0.0)
        r.write_data_to_sim()
        # settle onto the wheels before the clock starts
        for _ in range(int(0.3 * cfg.physics_hz)):
            self._physics_step(render=False)
        self.t = 0.0
        self._step_i = 0
        self._wheel_angle = 0.0
        self._throttle_q: deque = deque()     # (apply at t, value), per channel: separate delays
        self._steer_q: deque = deque()
        self._applied = Command()
        self._pending: list[DetectionFrame] = []
        self._imu_out = []
        self._crashed = False
        self._last_rgb, self._last_dbg, self._last_frame = None, None, None
        q = self._quat()
        self._q_prev = None
        self.imu_model.reset(0.0, q)
        self.encoder.reset(0.0, 0.0)
        self.detector.reset()
        self.progress = ProgressTracker(self.track, reverse=ep.reverse)
        self.progress.reset(x, y, 0.0)
        return self.car_info

    # ------------------------------------------------------------------------------------------------
    def step(self, cmd: Command) -> tuple[SensorFrame, Snapshot]:
        """Queue the algorithm's command, advance one control period, return what became available."""
        c = self.cfg.car
        cmd = cmd.clamped(c.max_throttle)
        self._throttle_q.append((self.t + c.throttle_delay_s, cmd.throttle))
        self._steer_q.append((self.t + c.steer_delay_s, cmd.steer))
        for _ in range(self._substeps):
            thr, steer = self._applied.throttle, self._applied.steer
            while self._throttle_q and self._throttle_q[0][0] <= self.t + 1e-9:
                thr = self._throttle_q.popleft()[1]
            while self._steer_q and self._steer_q[0][0] <= self.t + 1e-9:
                steer = self._steer_q.popleft()[1]
            self._applied = Command(thr, steer)
            self.car.apply(self._applied.throttle, self._applied.steer, disabled=self._crashed)
            render = self._step_i % self._cam_every == 0
            self._physics_step(render=render)
            self._step_i += 1
            self.t += self.dt
            self._sense(render)

        frames = [f for f in self._pending if f.t_available <= self.t + 1e-9]
        self._pending = [f for f in self._pending if f.t_available > self.t + 1e-9]
        sensors = SensorFrame(self.t, frames, self._imu_out, self.encoder.reading(self.t))
        self._imu_out = []
        status = self._status()
        return sensors, Snapshot(status, self._last_rgb, self._last_dbg, self._last_frame, self._applied)

    # ------------------------------------------------------------------------------------------------
    def _physics_step(self, render: bool) -> None:
        self.robot.write_data_to_sim()
        self.sim.step(render=False)
        if render:
            self.sim.render()
        self.robot.update(self.dt)
        self.imu.update(self.dt)
        self.contact.update(self.dt)

    def _quat(self):
        return tuple(float(v) for v in self.imu.data.quat_w[0].tolist())

    def _body_rate(self, q) -> np.ndarray:
        """Body-frame angular velocity from the orientation change over the last physics step.

        Not PhysX's reported angular velocity: with the USD's parallel (non-Ackermann) steering the
        tyres scrub in every turn and the solver's position corrections rotate the chassis without
        that rotation appearing in its velocity - integrated, the reported yaw rate comes out ~4 %
        short in a hairpin and drifts ~2 deg/s against a wall. A gyro measures the actual rotation."""
        q = np.asarray(q, dtype=np.float64)
        prev, self._q_prev = self._q_prev, q
        if prev is None:
            return np.zeros(3)
        w0, v0 = prev[0], -prev[1:]                                   # conj(prev) * q
        w1, v1 = q[0], q[1:]
        dw = w0 * w1 - v0 @ v1
        dv = w0 * v1 + w1 * v0 + np.cross(v0, v1)
        if dw < 0:
            dw, dv = -dw, -dv
        angle = 2.0 * math.atan2(np.linalg.norm(dv), dw)
        n = np.linalg.norm(dv)
        return dv / n * angle / self.dt if n > 1e-12 else np.zeros(3)

    def _sense(self, rendered: bool) -> None:
        d = self.imu.data
        quat = self._quat()
        gyro = self._body_rate(quat)
        sample = self.imu_model.step(self.t, gyro, d.lin_acc_b[0].cpu().numpy(), quat)
        if sample is not None:
            self._imu_out.append(sample)
        wv = float(self.robot.data.joint_vel[0, self.car.throttle_ids].mean())
        self._wheel_angle += wv * self.dt
        self.encoder.step(self.t, self._wheel_angle)

        if self.t > GRACE_S and not self._crashed:
            fm = self.contact.data.force_matrix_w          # [1, links, 1 filter, 3]
            if float(torch.linalg.vector_norm(fm[..., :2], dim=-1).max()) > self.cfg.episode.wall_force_thresh_n:
                self._crashed = True

        if rendered:
            self.camera.update(self.dt, force_recompute=True)
            view = self._object_view()
            frame, dbg = self.detector.process(self.t, view, yaw_rate=float(gyro[2]))
            self._last_dbg, self._last_frame = dbg, frame
            if frame is not None:
                self._pending.append(frame)
            if self.cfg.camera.render_rgb:
                self._last_rgb = self.camera.data.output["rgb"][0].cpu().numpy()

    # ------------------------------------------------------------------------------------------------
    def _object_view(self) -> ObjectView:
        """Per-object ground truth for this frame: projected outline + what the renderer drew."""
        cam, W, H = self.camera, self.cfg.camera.width_px, self.cfg.camera.height_px
        Wr, Hr = self.render_model.cam.width_px, self.render_model.cam.height_px
        seg = cam.data.output["instance_id_segmentation_fast"][0, ..., 0].to(torch.long)
        id_to_labels = cam.data.info.get("instance_id_segmentation_fast", {}).get("idToLabels", {})
        if len(id_to_labels) != self._lut_size:
            self._rebuild_lut(id_to_labels)
        n = self.track.n_objects
        ids = seg.clamp(0, self._lut.numel() - 1)
        obj = torch.where(seg < self._lut.numel(), self._lut[ids], torch.full_like(ids, -1))
        ys, xs = torch.nonzero(obj >= 0, as_tuple=True)
        o = obj[ys, xs]
        count = torch.bincount(o, minlength=n).cpu().numpy()
        box = torch.stack([
            torch.full((n,), Wr, device=self.device, dtype=torch.long).scatter_reduce(0, o, xs, "amin"),
            torch.full((n,), Hr, device=self.device, dtype=torch.long).scatter_reduce(0, o, ys, "amin"),
            torch.full((n,), -1, device=self.device, dtype=torch.long).scatter_reduce(0, o, xs, "amax") + 1,
            torch.full((n,), -1, device=self.device, dtype=torch.long).scatter_reduce(0, o, ys, "amax") + 1,
        ], 1).cpu().numpy().astype(np.float64)

        # Full outline of every object near the camera. Camera pose from the base_link body pose + the mount offset. (TiledCamera's own pos_w is not
        # refreshed from the physics pose with CPU PhysX, so it cannot be used here.)
        b = self.car.base_body
        R_base = quat_to_matrix(self.robot.data.body_quat_w[0, b].cpu().numpy())
        pos = self.robot.data.body_pos_w[0, b].cpu().numpy().astype(np.float64) + R_base @ self._cam_pos_b
        R = R_base @ self._cam_rot_b                                          # camera (x right, y down, z fwd) -> world
        centers = np.c_[self.track.xy, self.track.z0 + self.track.height / 2]
        near = np.linalg.norm(centers - pos, axis=1) < self.cfg.detector.max_range_m
        cand = np.nonzero(near | (count > 0))[0]
        pc = (self._outline[cand] - pos) @ R                                      # [M, P, 3] camera frame
        front = (pc[..., 2] > 0.05).all(1)
        cand, pc = cand[front], pc[front]
        u, v = self.lens.project_cam(pc)                                     # detector (lens) pixels
        amodal = np.stack([u.min(1), v.min(1), u.max(1), v.max(1)], 1)
        overlap = (amodal[:, 2] > 0) & (amodal[:, 0] < W) & (amodal[:, 3] > 0) & (amodal[:, 1] < H)
        cand, amodal, pc = cand[overlap], amodal[overlap], pc[overlap]
        # visibility is measured in the render: pixels drawn / area of the full silhouette there
        ur, vr = self.render_model.project_cam(pc)
        area = np.array([_hull_area(np.stack([a, b], 1)) for a, b in zip(ur, vr)])
        depth = np.linalg.norm(centers[cand] - pos, axis=1)
        return ObjectView(cand, amodal, area, count[cand].astype(np.float64), self._render_box_to_lens(box[cand]), depth)

    def _render_box_to_lens(self, boxes: np.ndarray) -> np.ndarray:
        """Render-pixel boxes [M, 4] -> boxes around the same region in lens pixels (border samples)."""
        if not len(boxes):
            return boxes
        t = np.linspace(0.0, 1.0, 5)
        x1, y1, x2, y2 = (boxes[:, i:i + 1] for i in range(4))
        bu = np.hstack([x1 + (x2 - x1) * t, x1 + (x2 - x1) * t, np.repeat(x1, 5, 1), np.repeat(x2, 5, 1)])
        bv = np.hstack([np.repeat(y1, 5, 1), np.repeat(y2, 5, 1), y1 + (y2 - y1) * t, y1 + (y2 - y1) * t])
        u, v = self.lens.project_cam(self.render_model.ray_cam(bu, bv))
        return np.stack([u.min(1), v.min(1), u.max(1), v.max(1)], 1)

    def _rebuild_lut(self, id_to_labels: dict) -> None:
        pairs = []
        for k, label in id_to_labels.items():          # {"17": "/World/Track/Cylinder_005", ...}
            name = str(label).rstrip("/").split("/")[-1]
            if name in self.track.name_to_id:
                pairs.append((int(k), self.track.name_to_id[name]))
        size = max([k for k, _ in pairs] + [0]) + 1
        lut = torch.full((size,), -1, dtype=torch.long, device=self.device)
        for k, v in pairs:
            lut[k] = v
        self._lut, self._lut_size = lut, len(id_to_labels)

    def _status(self) -> Status:
        d = self.robot.data
        x, y = float(d.root_pos_w[0, 0]), float(d.root_pos_w[0, 1])
        q = d.root_quat_w[0].tolist()
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        up_z = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
        # report the car-frame origin (rear axle), not base_link
        rx = x + math.cos(yaw) * self.car.geom["rear_x"]
        ry = y + math.sin(yaw) * self.car.geom["rear_x"]
        lateral = self.progress.update(rx, ry, self.t)
        p, ep = self.progress, self.cfg.episode
        st = Status(self.t, rx, ry, yaw, float(d.root_lin_vel_b[0, 0]), float(d.root_ang_vel_w[0, 2]),
                    p.progress, lateral, p.laps, list(p.lap_times))
        for cond, reason in ((p.laps >= ep.laps, "laps"), (self._crashed, "crash"), (up_z < 0.3, "flip"),
                             (p.progress < p.best - ep.wrong_way_m, "wrong_way"),
                             (p.since_progress(self.t) > ep.stuck_time_s, "stuck"), (self.t >= ep.max_time_s, "timeout")):
            if cond:
                st.done, st.reason = True, reason
                break
        return st

    def chase_camera(self, st: Status) -> None:
        if self.gui:
            back, up = 1.2, 0.7
            self.sim.set_camera_view([st.x - back * math.cos(st.yaw), st.y - back * math.sin(st.yaw), up],
                                     [st.x + 1.5 * math.cos(st.yaw), st.y + 1.5 * math.sin(st.yaw), 0.0])


# ------------------------------------------------------------------------------------------------------
def _covering_pinhole(lens: CameraModel, scale: float) -> CameraModel:
    """Pinhole camera with the lens camera's pose, wide enough to contain every ray of the lens frame."""
    c = lens.cam
    W, H = c.width_px, c.height_px
    t = np.linspace(0.0, 1.0, 64)
    bu = np.r_[t * W, t * W, np.zeros(64), np.full(64, W)]
    bv = np.r_[np.zeros(64), np.full(64, H), t * H, t * H]
    d = lens.ray_cam(bu, bv)
    if (d[:, 2] <= 0.05).any():
        raise ValueError("lens field of view too wide to render with a pinhole camera (> ~170 deg)")
    half_w = max(np.abs(d[:, 0] / d[:, 2]).max(), np.abs(d[:, 1] / d[:, 2]).max() * W / H) * 1.01
    Wr, Hr = int(round(W * scale / 2)) * 2, int(round(H * scale / 2)) * 2
    return CameraModel(CameraInfo(Wr, Hr, 2 * math.degrees(math.atan(half_w)), c.x_m, c.y_m, c.z_m,
                                  c.pitch_deg, c.yaw_deg, "pinhole"))


def lens_remap(lens: CameraModel, render: CameraModel) -> tuple[np.ndarray, np.ndarray]:
    """cv2.remap maps that turn a render image into the lens (detector) image, for the viewer."""
    W, H = lens.cam.width_px, lens.cam.height_px
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    ru, rv = render.project_cam(lens.ray_cam(u, v))
    return (ru - 0.5).astype(np.float32), (rv - 0.5).astype(np.float32)


def _ros_camera_quat(pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    """(w, x, y, z) of the ROS-convention camera (x right, y down, z forward) relative to base_link."""
    return _matrix_to_quat_wxyz(camera_rotation(pitch_deg, yaw_deg))


def _matrix_to_quat_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return tuple(q)


def _object_outlines(stage, track: TrackMap) -> np.ndarray:
    """[N, SUPPORT_DIRS, 3] world points of every object's mesh that are extreme in some direction.

    Any shape the .blend uses (cylinder, sphere, apple mesh) projects to a silhouette whose box and
    convex hull are those of these support points, so the rest of the vertices can be dropped."""
    from pxr import UsdGeom

    k = np.arange(SUPPORT_DIRS) + 0.5                       # Fibonacci sphere of directions
    phi, theta = np.arccos(1 - 2 * k / SUPPORT_DIRS), math.pi * (1 + 5 ** 0.5) * k
    dirs = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    cache, out = UsdGeom.XformCache(), []
    for name in track.names:
        prim = stage.GetPrimAtPath(f"/World/Track/{name}")
        pts = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
        m = np.array(cache.GetLocalToWorldTransform(prim), dtype=np.float64)   # row-vector convention
        world = pts @ m[:3, :3] + m[3, :3]
        out.append(world[np.argmax(world @ dirs.T, axis=0)])
    return np.array(out)


def _hull_area(p: np.ndarray) -> float:
    """Area of the convex hull of 2D points."""
    return float(cv2.contourArea(cv2.convexHull(p.astype(np.float32))))
