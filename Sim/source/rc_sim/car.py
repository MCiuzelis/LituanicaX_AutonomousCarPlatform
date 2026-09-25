"""The simulated car: MuSHR nano v2 USD re-dimensioned to the real car, plus its drivetrain model.

Carried over from the calibrated RL environment (old repo, Training/source/cone_track/env.py) with
everything policy-related removed. See config.CarCfg for the numbers and where they came from.
Import only after Isaac Sim has been launched.
"""
from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg

from .config import CAR_USD, CarCfg

ROBOT_PATH = "/World/Robot"
ROOT_LINK = f"{ROBOT_PATH}/mushr_nano"
BASE_LINK = f"{ROOT_LINK}/base_link"
USD_WHEELBASE_M = 0.1385 + 0.158        # axle spacing authored in the USD, unscaled
WHEEL_LINKS = ("front_left_wheel_link", "front_right_wheel_link", "back_left_wheel_link", "back_right_wheel_link")
THROTTLE_JOINTS = ("back_left_wheel_throttle", "back_right_wheel_throttle",
                   "front_left_wheel_throttle", "front_right_wheel_throttle")
STEER_JOINTS = ("front_left_wheel_steer", "front_right_wheel_steer")


def robot_cfg(cfg: CarCfg) -> ArticulationCfg:
    s = cfg.robot_scale
    return ArticulationCfg(
        prim_path=ROBOT_PATH,
        spawn=sim_utils.UsdFileCfg(
            usd_path=CAR_USD,
            scale=(s, s, s),
            activate_contact_sensors=True,
            # no MassPropertiesCfg: it would stamp one mass on all 17 bodies. The USD's own
            # distribution is scaled to cfg.mass_kg in Car.after_sim_start().
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, max_linear_velocity=10000.0, max_angular_velocity=100000.0,
                max_depenetration_velocity=1.0, max_contact_impulse=5.0, enable_gyroscopic_forces=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=4,
                solver_velocity_iteration_count=0, sleep_threshold=0.005, stabilization_threshold=0.001),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
        actuators={
            "steering": ImplicitActuatorCfg(joint_names_expr=list(STEER_JOINTS), stiffness=100.0, damping=10.0,
                                            velocity_limit_sim=cfg.servo_rate_rad_s, effort_limit_sim=3.2),
            # pure torque control: the DC-motor curve is computed in Car.apply()
            "throttle": ImplicitActuatorCfg(joint_names_expr=[".*_wheel_throttle"], stiffness=0.0, damping=0.0,
                                            friction=0.0, effort_limit_sim=5.0, velocity_limit_sim=1e9),
            # suspension welded at zero travel (the USD's compliant drives bleed energy)
            "suspension": ImplicitActuatorCfg(joint_names_expr=[".*_wheel_suspension"],
                                              stiffness=cfg.suspension_stiffness, damping=0.0, friction=0.0,
                                              effort_limit_sim=1e9, velocity_limit_sim=1e9),
        },
    )


class Car:
    """Spawns the articulation, fixes its geometry, and turns Commands into joint targets."""

    def __init__(self, cfg: CarCfg, stage):
        self.cfg = cfg
        self.robot = Articulation(robot_cfg(cfg))
        self._fix_geometry(stage)
        self.geom = self._measure_geometry(stage)

    # ---- USD edits (before the simulation starts) ------------------------------------------------
    def _fix_geometry(self, stage) -> None:
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

        c, s = self.cfg, self.cfg.robot_scale
        # 1) wheelbase: the uniform spawn scale that makes the wheels 74 mm also made the wheelbase
        #    224 mm instead of the real 187 mm. Scale every axle position about base_link.
        factor = (c.wheelbase_m / s) / USD_WHEELBASE_M
        links = {"front": ("_wheel_suspension_link", "_wheel_steer_link", "_wheel_link"),
                 "back": ("_wheel_suspension_link", "_wheel_link")}
        for corner in ("front_left", "front_right", "back_left", "back_right"):
            joint = stage.GetPrimAtPath(f"{BASE_LINK}/{corner}_wheel_suspension")
            attr = joint.GetAttribute("physics:localPos0")
            if not joint.IsValid() or not attr.IsValid():
                raise RuntimeError(f"car USD changed shape: {corner} suspension joint not found")
            p = attr.Get()
            attr.Set(type(p)(p[0] * factor, p[1], p[2]))
            for suffix in links[corner.split("_")[0]]:
                t = stage.GetPrimAtPath(f"{ROOT_LINK}/{corner}{suffix}").GetAttribute("xformOp:translate")
                v = t.Get()
                t.Set(type(v)(v[0] * factor, v[1], v[2]))
            # suspension drives from the USD would fight the welded actuator
            for prop in list(joint.GetProperties()):
                if "stiffness" in prop.GetName() or "damping" in prop.GetName():
                    joint.GetAttribute(prop.GetName()).Block()

        # 2) wheel collider: the enabled collider is a Cylinder authored 7 % oversize (39.6 mm)
        root = stage.GetPrimAtPath(ROOT_LINK)
        r_local = c.wheel_radius_m / s
        rubber = stage.DefinePrim("/World/PhysicsMaterials/WheelRubber", "Material")
        mat = UsdPhysics.MaterialAPI.Apply(rubber)
        mat.CreateStaticFrictionAttr().Set(1.0)
        mat.CreateDynamicFrictionAttr().Set(1.0)
        mat.CreateRestitutionAttr().Set(0.05)
        for prim in Usd.PrimRange(root):
            if prim.GetParent().GetName() not in WHEEL_LINKS:
                continue
            if prim.GetTypeName() == "Cylinder":
                sc = prim.GetAttribute("xformOp:scale")
                radial = float(sc.Get()[0]) if sc and sc.Get() is not None else 1.0
                UsdGeom.Cylinder(prim).GetRadiusAttr().Set(r_local / radial)
            # Same bindings as the calibrated RL sim (meshes only), so its drivetrain fit still holds.
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.MeshCollisionAPI.Apply(prim).GetApproximationAttr().Set("convexHull")
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    UsdShade.Material(rubber), UsdShade.Tokens.strongerThanDescendants, "physics")

    def _measure_geometry(self, stage) -> dict:
        """Where the car frame (rear-axle centre on the ground) sits relative to base_link, in metres."""
        from pxr import UsdGeom

        cache = UsdGeom.XformCache()
        base = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(BASE_LINK))
        base_inv = base.GetInverse()
        rel = {}
        for name in WHEEL_LINKS:
            w = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{ROOT_LINK}/{name}"))
            p = base_inv.Transform(w.ExtractTranslation())
            rel[name] = [v * self.cfg.robot_scale for v in p]      # base_link frame is scaled
        rear_x = 0.5 * (rel["back_left_wheel_link"][0] + rel["back_right_wheel_link"][0])
        front_x = 0.5 * (rel["front_left_wheel_link"][0] + rel["front_right_wheel_link"][0])
        wheel_z = sum(r[2] for r in rel.values()) / 4
        track = abs(rel["back_left_wheel_link"][1] - rel["back_right_wheel_link"][1])
        # car-frame origin expressed in base_link coordinates (metres)
        return {"rear_x": rear_x, "ground_z": wheel_z - self.cfg.wheel_radius_m,
                "wheelbase": front_x - rear_x, "track": track}

    def car_to_base(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Car-frame point (m) -> base_link-frame point (m)."""
        return x + self.geom["rear_x"], y, z + self.geom["ground_z"]

    # ---- after sim.reset() -----------------------------------------------------------------------
    def after_sim_start(self) -> None:
        r = self.robot
        self.throttle_ids = torch.tensor(r.find_joints(list(THROTTLE_JOINTS), preserve_order=True)[0],
                                         dtype=torch.long, device=r.device)
        self.steer_ids = r.find_joints(list(STEER_JOINTS), preserve_order=True)[0]
        self.base_body = r.find_bodies("base_link")[0][0]
        pv = r.root_physx_view
        masses = pv.get_masses()
        native = masses[0].sum().item()
        k = self.cfg.mass_kg / native
        ids = torch.arange(r.num_instances, dtype=torch.int, device="cpu")
        pv.set_masses(masses * k, ids)
        pv.set_inertias(pv.get_inertias() * k, ids)
        c = self.cfg
        self.omega_noload = c.noload_speed_full_duty_m_s / c.wheel_radius_m
        self.rolling_torque = c.rolling_resistance_m_s2 * c.wheel_radius_m * c.mass_kg / 4.0
        self.understeer = math.radians(c.understeer_deg_per_g) / 9.81
        print(f"[car] mass {native:.3f} kg (USD) -> {c.mass_kg:.3f} kg | wheelbase {self.geom['wheelbase'] * 1000:.0f} mm"
              f" | track {self.geom['track'] * 1000:.0f} mm | base_link {-self.geom['ground_z'] * 1000:.1f} mm above ground,"
              f" {-self.geom['rear_x'] * 1000:.1f} mm ahead of the rear axle")

    def apply(self, throttle: float, steer: float, disabled: bool = False) -> None:
        """Command (Pi5 convention: throttle -1..1, steer + = right) -> wheel torques + steer target."""
        c, r = self.cfg, self.robot
        omega = r.data.joint_vel[:, self.throttle_ids].clamp(min=0.0)            # [1, 4] rad/s
        if throttle > c.esc_deadband:
            duty = (throttle - c.esc_deadband) / (1.0 - c.esc_deadband) * c.esc_duty_per_throttle
            torque = c.stall_torque_full_duty_nm * (duty - omega / self.omega_noload).clamp(0.0, 1.0)
        elif throttle < -c.esc_deadband:
            level = min(1.0, -throttle)
            torque = -(c.brake_torque_base_nm + c.brake_torque_full_nm * level) * (omega / c.brake_release_omega).clamp(0.0, 1.0)
        else:
            torque = torch.zeros_like(omega)
        torque = torque - self.rolling_torque * (omega / c.rolling_release_omega).clamp(0.0, 1.0)
        if disabled:
            torque = torch.zeros_like(torque)
        r.set_joint_effort_target(torque, joint_ids=self.throttle_ids)

        # USD steer joints: + = left; the car's Drive.steer(): + = right
        delta = -steer * math.radians(c.max_steer_deg)
        if c.understeer_deg_per_g > 0.0 and delta != 0.0:
            a_lat = abs(float(r.data.root_lin_vel_b[0, 0]) * float(r.data.root_ang_vel_w[0, 2]))
            loss = min(self.understeer * a_lat, abs(delta) * c.understeer_max_reduction)
            delta -= math.copysign(loss, delta)
        if disabled:
            delta = 0.0
        r.set_joint_position_target(torch.full((1, 2), delta, device=r.device), joint_ids=self.steer_ids)
