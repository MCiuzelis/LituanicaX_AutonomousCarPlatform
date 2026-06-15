import os
import re

file_path = "/home/matasciuzelis/Documents/LituanicaX_InteligentRobotCompetition/Training/source/tasks/cone_track/env.py"

with open(file_path, "r") as f:
    content = f.read()

# 1. Replace track.usdc with stripes.usdc
content = content.replace('TRACK_USD = os.path.join(_EXPORT_DIR, "Track.usdc")', 'TRACK_USD = os.path.join(_EXPORT_DIR, "stripes.usdc")')

# 2. Replace _TOF_ optics with _RGB_ optics
old_optics = '''_TOF_DIAG_FOV_DEG = 70.0
_TOF_DIAG_PIXELS = math.sqrt(240**2 + 180**2)  # 300 px
_TOF_FOCAL_PX = _TOF_DIAG_PIXELS / 2.0 / math.tan(math.radians(_TOF_DIAG_FOV_DEG / 2.0))
_TOF_HFOV_DEG = 2.0 * math.degrees(math.atan2(120.0, _TOF_FOCAL_PX))  # ≈ 58.52°
_TOF_VFOV_DEG = 2.0 * math.degrees(math.atan2(90.0, _TOF_FOCAL_PX))  # ≈ 45.57°
_TOF_FOCAL_MM = 4.0  # reference focal length (mm); aperture ratio is what matters
_TOF_H_APERTURE_MM = 2.0 * _TOF_FOCAL_MM * math.tan(math.radians(_TOF_HFOV_DEG / 2.0))
_TOF_V_APERTURE_MM = 2.0 * _TOF_FOCAL_MM * math.tan(math.radians(_TOF_VFOV_DEG / 2.0))'''

new_optics = '''_RGB_FOCAL_MM = 2.0
_RGB_H_APERTURE_MM = 2.0 * _RGB_FOCAL_MM * math.tan(math.radians(120.0 / 2.0))
_RGB_V_APERTURE_MM = 2.0 * _RGB_FOCAL_MM * math.tan(math.radians(66.7 / 2.0))'''

content = content.replace(old_optics, new_optics)

# 3. Replace camera_left_cfg and camera_right_cfg
old_cameras = '''    camera_left_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/mushr_nano/base_link/camera_left",
        update_period=8.0
        / 240.0,  # render only at policy steps (decimation=8, sim=240Hz)
        # Render at the sensor's native 240x180 to match the real ArduCam TOF.
        # All noise effects are applied at this full resolution before cropping
        # and downsampling, mirroring how the real camera produces noisy depth.
        height=180,
        width=240,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_TOF_FOCAL_MM,
            horizontal_aperture=_TOF_H_APERTURE_MM,
            vertical_aperture=_TOF_V_APERTURE_MM,
            clipping_range=(0.01, 5.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(
                0.185,
                0.04,
                -0.008,
            ),  # 122 mm fwd, 50 mm left, 70 mm up from base_link
            rot=(-0.6029, 0.6029, -0.3695, 0.3695),  # 27° yaw left, no pitch
            convention="ros",
        ),
        debug_vis=False,
    )
    camera_right_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/mushr_nano/base_link/camera_right",
        update_period=8.0
        / 240.0,  # render only at policy steps (decimation=8, sim=240Hz)
        height=180,
        width=240,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_TOF_FOCAL_MM,
            horizontal_aperture=_TOF_H_APERTURE_MM,
            vertical_aperture=_TOF_V_APERTURE_MM,
            clipping_range=(0.01, 5.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(
                0.185,
                -0.04,
                -0.008,
            ),  # 122 mm fwd, 50 mm right, 70 mm up from base_link
            rot=(-0.3695, 0.3695, -0.6029, 0.6029),  # 27° yaw right, no pitch
            convention="ros",
        ),
        debug_vis=False,
    )'''

new_camera = '''    camera_center_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/mushr_nano/base_link/camera_center",
        update_period=1.0 / 120.0,
        height=480,
        width=848,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_RGB_FOCAL_MM,
            horizontal_aperture=_RGB_H_APERTURE_MM,
            vertical_aperture=_RGB_V_APERTURE_MM,
            clipping_range=(0.01, 100.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(
                0.185,
                0.0,
                -0.008,
            ),
            rot=(-0.5, 0.5, -0.5, 0.5),  # No rotation, looking forward
            convention="ros",
        ),
        debug_vis=False,
    )'''

content = content.replace(old_cameras, new_camera)

# 4. Modify init calculations for observation space
old_init_cam = '''        cam_h = cfg.camera_left_cfg.height  # 180
        cam_w = cfg.camera_left_cfg.width  # 240
        remaining = max(
            0.0, 1.0 - cfg.depth_crop_top_fraction - cfg.depth_crop_bottom_fraction
        )
        cropped_h = max(1, int(cam_h * remaining))
        cropped_w = cam_w

        self._policy_depth_w = cfg.policy_depth_width
        if cfg.policy_depth_height < 0:
            self._policy_depth_h = max(
                1, round(cfg.policy_depth_width * cropped_h / cropped_w)
            )
        else:
            self._policy_depth_h = cfg.policy_depth_height

        depth_obs_size = self._policy_depth_h * self._policy_depth_w

        extra_obs = 0
        if cfg.include_wheel_vel:
            extra_obs += 1  # scalar: avg wheel surface speed normalized to [0, 1]
        cfg.observation_space = depth_obs_size * 2 + extra_obs  # left + right cameras'''

new_init_cam = '''        cam_h = cfg.camera_center_cfg.height
        cam_w = cfg.camera_center_cfg.width
        cropped_h = cam_h
        cropped_w = cam_w

        self._policy_depth_w = cfg.policy_depth_width
        if cfg.policy_depth_height < 0:
            self._policy_depth_h = max(
                1, round(cfg.policy_depth_width * cropped_h / cropped_w)
            )
        else:
            self._policy_depth_h = cfg.policy_depth_height

        vision_obs_size = self._policy_depth_h * self._policy_depth_w

        extra_obs = 0
        if cfg.include_wheel_vel:
            extra_obs += 1

        cfg.observation_space = vision_obs_size + extra_obs  # 1 camera'''

content = content.replace(old_init_cam, new_init_cam)

# 5. Modify print statement after init
old_print = '''        print(
            f"[ConeTrackEnv] Depth obs per camera: raw {cam_w}×{cam_h} "
            f"→ crop {cropped_w}×{cropped_h} "
            f"→ policy {self._policy_depth_w}×{self._policy_depth_h} "
            f"= {depth_obs_size} px  |  2 cameras  |  "
            f"total obs = {cfg.observation_space}"
        )'''

new_print = '''        print(
            f"[ConeTrackEnv] Vision obs: raw {cam_w}×{cam_h} "
            f"→ policy {self._policy_depth_w}×{self._policy_depth_h} "
            f"= {vision_obs_size} px  |  1 camera  |  "
            f"total obs = {cfg.observation_space}"
        )'''

content = content.replace(old_print, new_print)

# 6. Change depth caches to vision cache
old_caches = '''        # Depth observation caches (policy resolution after crop + downsample)
        _cam_flat = self._policy_depth_h * self._policy_depth_w
        self._camera_left_obs = torch.zeros(
            self.num_envs, _cam_flat, device=self.device
        )
        self._camera_right_obs = torch.zeros(
            self.num_envs, _cam_flat, device=self.device
        )
        self._last_camera_update_step = -1'''

new_caches = '''        # Vision observation caches (policy resolution after color filtering + downsample)
        _cam_flat = self._policy_depth_h * self._policy_depth_w
        self._camera_center_obs = torch.zeros(
            self.num_envs, _cam_flat, device=self.device
        )
        self._last_camera_update_step = -1'''

content = content.replace(old_caches, new_caches)

# 7. Setup scene camera updates
old_scene_cam = '''        self.camera_left = TiledCamera(self.cfg.camera_left_cfg)
        self.camera_right = TiledCamera(self.cfg.camera_right_cfg)'''

new_scene_cam = '''        self.camera_center = TiledCamera(self.cfg.camera_center_cfg)'''

content = content.replace(old_scene_cam, new_scene_cam)

old_scene_sensors = '''        self.scene.sensors["camera_left"] = self.camera_left
        self.scene.sensors["camera_right"] = self.camera_right'''

new_scene_sensors = '''        self.scene.sensors["camera_center"] = self.camera_center'''

content = content.replace(old_scene_sensors, new_scene_sensors)

# 8. Process single depth to process single RGB
old_process = '''    def _process_single_depth(self, camera) -> torch.Tensor:
        """Shared depth pipeline for one TiledCamera → [N, policy_h*policy_w].

        Noise effects are applied at the camera's full 240×180 resolution before
        cropping and downsampling, matching the real ArduCam TOF sensor pipeline.
        """
        import torch.nn.functional as F

        depth_raw = camera.data.output["distance_to_image_plane"]  # [N, H, W, 1]
        depth = depth_raw[..., 0]  # [N, H, W]
        depth = torch.where(
            torch.isfinite(depth),
            depth,
            torch.tensor(self.cfg.depth_max_m, device=depth.device),
        )

        # Edge weights computed on the FULL-RESOLUTION CLEAN signal so that real
        # depth discontinuities (not noise spikes) drive the mixed-pixel scatter.
        if self.cfg.depth_edge_scatter_prob > 0.0:
            _d4 = depth.unsqueeze(1)  # [N,1,H,W]
            _px = F.pad(_d4, (1, 1, 0, 0), mode="replicate")  # [N,1,H,W+2]
            _py = F.pad(_d4, (0, 0, 1, 1), mode="replicate")  # [N,1,H+2,W]
            _dx = (_px[:, :, :, 2:] - _px[:, :, :, :-2]) * 0.5  # centred ∂/∂x
            _dy = (_py[:, :, 2:, :] - _py[:, :, :-2, :]) * 0.5  # centred ∂/∂y
            _edge_w = (
                (_dx.abs() + _dy.abs()).squeeze(1)  # L1 grad [N,H,W]
                / self.cfg.depth_edge_gradient_threshold_m
            ).clamp(0.0, 1.0)

        # ── Noise passes at full 240×180 sensor resolution ────────────────────
        if self.cfg.depth_noise_sigma_m > 0.0:
            depth = depth + torch.randn_like(depth) * self.cfg.depth_noise_sigma_m

        if self.cfg.depth_invalid_pixel_prob > 0.0:
            invalid = torch.rand_like(depth) < self.cfg.depth_invalid_pixel_prob
            depth = depth.masked_fill(invalid, self.cfg.depth_max_m)

        if self.cfg.depth_color_noise_prob > 0.0:
            color_mask = torch.rand_like(depth) < self.cfg.depth_color_noise_prob
            depth = torch.where(
                color_mask,
                torch.rand_like(depth) * self.cfg.depth_max_m,
                depth,
            )

        if self.cfg.depth_edge_scatter_prob > 0.0:
            scatter_mask = torch.rand_like(depth) < (
                self.cfg.depth_edge_scatter_prob * _edge_w
            )
            scatter_vals = torch.where(
                torch.rand_like(depth) > 0.5,
                torch.full_like(depth, self.cfg.depth_max_m),
                torch.zeros_like(depth),
            )
            depth = torch.where(scatter_mask, scatter_vals, depth)

        if (
            self.cfg.depth_edge_blur_sigma_m > 0.0
            and self.cfg.depth_edge_scatter_prob > 0.0
        ):
            depth = depth + torch.randn_like(depth) * (
                self.cfg.depth_edge_blur_sigma_m * _edge_w
            )

        # ── Crop AFTER noise (removes top/bottom rows at full resolution) ─────
        h = depth.shape[1]
        top = int(self.cfg.depth_crop_top_fraction * h)
        bot = h - int(self.cfg.depth_crop_bottom_fraction * h)
        depth = depth[:, top:bot, :]  # [N, H', W]

        depth = depth.clamp(0.01, self.cfg.depth_max_m)
        depth_norm = 1.0 - depth / self.cfg.depth_max_m
        depth_4d = F.interpolate(
            depth_norm.unsqueeze(1),
            size=(self._policy_depth_h, self._policy_depth_w),
            mode="bilinear",
            align_corners=False,
        )
        return depth_4d.squeeze(1).reshape(self.num_envs, -1)'''

new_process = '''    def _process_single_rgb(self, camera) -> torch.Tensor:
        """Shared vision pipeline for RGB camera → [N, policy_h*policy_w]."""
        import cv2
        import numpy as np
        import torch.nn.functional as F

        # rgb output is [N, H, W, 4] uint8
        rgb_tensor = camera.data.output["rgb"][..., :3]
        N, H, W, _ = rgb_tensor.shape
        
        # move to CPU for cv2 processing
        rgb_np = rgb_tensor.cpu().numpy().astype(np.uint8)
        binary_masks = np.zeros((N, H, W), dtype=np.float32)

        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])

        for i in range(N):
            img_rgb = rgb_np[i]
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            
            mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
            binary_masks[i] = mask.astype(np.float32) / 255.0
            
        mask_tensor = torch.from_numpy(binary_masks).to(camera.device)
        
        mask_4d = F.interpolate(
            mask_tensor.unsqueeze(1),
            size=(self._policy_depth_h, self._policy_depth_w),
            mode="bilinear",
            align_corners=False,
        )
        return mask_4d.squeeze(1).reshape(self.num_envs, -1)'''

content = content.replace(old_process, new_process)

# 9. Update caches call
old_update = '''    def _update_depth_cache(self) -> None:
        """Process both depth cameras and write self._camera_left_obs / _camera_right_obs."""
        current_step = int(self.episode_length_buf.max().item())
        if self._last_camera_update_step == current_step:
            return
        self._last_camera_update_step = current_step
        self._camera_left_obs = self._process_single_depth(self.camera_left)
        self._camera_right_obs = self._process_single_depth(self.camera_right)'''

new_update = '''    def _update_depth_cache(self) -> None:
        """Process center camera and write self._camera_center_obs."""
        current_step = int(self.episode_length_buf.max().item())
        if self._last_camera_update_step == current_step:
            return
        self._last_camera_update_step = current_step
        self._camera_center_obs = self._process_single_rgb(self.camera_center)'''

content = content.replace(old_update, new_update)

# 10. Update observation
old_obs = '''        parts = [self._camera_left_obs, self._camera_right_obs]'''

new_obs = '''        parts = [self._camera_center_obs]'''

content = content.replace(old_obs, new_obs)

with open(file_path, "w") as f:
    f.write(content)

print("Modification done!")
