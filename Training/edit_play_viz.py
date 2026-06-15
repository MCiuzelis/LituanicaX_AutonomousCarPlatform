import os

# --- Edit play.py ---
play_path = "/home/matasciuzelis/Documents/LituanicaX_InteligentRobotCompetition/Training/play.py"
with open(play_path, "r") as f:
    play_content = f.read()

play_content = play_content.replace('camera_left_cfg', 'camera_center_cfg')

old_depth_bgr = '''    def _depth_to_bgr_pair(camera, suffix: str) -> None:
        """Process one camera → write raw_{suffix}.png and policy_{suffix}.png."""
        depth_tensor = camera.data.output["distance_to_image_plane"]  # [1,H,W,1]
        depth_np = depth_tensor[0, :, :, 0].cpu().numpy()
        depth_max = _mushr_env.cfg.depth_max_m
        depth_np = np.where(np.isfinite(depth_np), depth_np, depth_max)
        raw_norm = 1.0 - np.clip(depth_np / depth_max, 0.0, 1.0)
        raw_bgr  = cv2.applyColorMap(
            np.clip(raw_norm * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        _write_atomic(os.path.join(_fpv_out_dir, f"raw_{suffix}.png"), raw_bgr)

        # Policy window: exact same pipeline as training (_process_single_depth).
        pol_flat = _mushr_env._process_single_depth(camera)   # [1, ph*pw], values [0,1]
        pol_2d   = pol_flat[0].reshape(
            _mushr_env._policy_depth_h, _mushr_env._policy_depth_w,
        ).cpu().numpy()
        pol_bgr  = cv2.applyColorMap(
            np.clip(pol_2d * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        _write_atomic(os.path.join(_fpv_out_dir, f"policy_{suffix}.png"), pol_bgr)

    def _fpv_step() -> None:
        """Grab depth frames from both cameras, false-colour them, write PNGs."""
        _depth_to_bgr_pair(_mushr_env.camera_left,  "left")
        _depth_to_bgr_pair(_mushr_env.camera_right, "right")
        if _fpv_writer is not None:
            # Write left raw frame to video for recording
            depth_tensor = _mushr_env.camera_left.data.output["distance_to_image_plane"]
            depth_np = depth_tensor[0, :, :, 0].cpu().numpy()
            depth_max = _mushr_env.cfg.depth_max_m
            depth_np = np.where(np.isfinite(depth_np), depth_np, depth_max)
            raw_norm = 1.0 - np.clip(depth_np / depth_max, 0.0, 1.0)
            raw_bgr  = cv2.applyColorMap(
                np.clip(raw_norm * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            _fpv_writer.write(raw_bgr)'''

new_vision_bgr = '''    def _vision_to_bgr(camera, suffix: str) -> None:
        """Process one camera → write raw_{suffix}.png and policy_{suffix}.png."""
        rgb_tensor = camera.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
        raw_bgr = cv2.cvtColor(rgb_tensor, cv2.COLOR_RGB2BGR)
        _write_atomic(os.path.join(_fpv_out_dir, f"raw_{suffix}.png"), raw_bgr)

        # Policy window
        pol_flat = _mushr_env._process_single_rgb(camera)
        pol_2d   = pol_flat[0].reshape(
            _mushr_env._policy_depth_h, _mushr_env._policy_depth_w,
        ).cpu().numpy()
        pol_bgr  = cv2.cvtColor((pol_2d * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        _write_atomic(os.path.join(_fpv_out_dir, f"policy_{suffix}.png"), pol_bgr)

    def _fpv_step() -> None:
        """Grab vision frame from center camera, write PNGs."""
        _vision_to_bgr(_mushr_env.camera_center, "center")
        if _fpv_writer is not None:
            rgb_tensor = _mushr_env.camera_center.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
            raw_bgr = cv2.cvtColor(rgb_tensor, cv2.COLOR_RGB2BGR)
            _fpv_writer.write(raw_bgr)'''

play_content = play_content.replace(old_depth_bgr, new_vision_bgr)

with open(play_path, "w") as f:
    f.write(play_content)

# --- Edit visualize.py ---
viz_path = "/home/matasciuzelis/Documents/LituanicaX_InteligentRobotCompetition/Training/visualize.py"
with open(viz_path, "r") as f:
    viz_content = f.read()

viz_content = viz_content.replace('camera_left_cfg', 'camera_center_cfg')

old_viz_write = '''    def _write_camera(cam_obj, suffix: str) -> None:
        """Write raw (clean) and policy (fully-noised) depth frames to disk.

        Raw window   — direct Isaac Sim sensor output, no noise applied, full resolution.
        Policy window — calls mushr_env._process_single_depth() which runs the IDENTICAL
                        pipeline the policy uses (crop → edge detection → all noise passes
                        → clamp → normalise → downsample), then upscales for display.
        """
        # ── Raw depth: unprocessed sensor output ──────────────────────────────
        raw_4d   = cam_obj.data.output["distance_to_image_plane"]  # [1, H, W, 1]
        depth_np = raw_4d[0, :, :, 0].cpu().numpy()
        depth_np = np.where(np.isfinite(depth_np), depth_np, depth_max)
        beyond   = depth_np >= depth_max
        raw_norm = 1.0 - np.clip(depth_np / depth_max, 0.0, 1.0)
        raw_bgr  = _depth_to_colormap(raw_norm)
        raw_bgr[beyond] = (0, 0, 0)            # out-of-range → black (matches real ArduCam)
        _write_atomic(os.path.join(out_dir, f"raw_{suffix}.png"), raw_bgr)

        # ── Policy depth: exact env pipeline ──────────────────────────────────
        pol_flat = mushr_env._process_single_depth(cam_obj)       # [1, ph*pw], values [0,1]
        pol_2d   = pol_flat[0].reshape(
            mushr_env._policy_depth_h, mushr_env._policy_depth_w,
        ).cpu().numpy()
        pol_bgr  = _depth_to_colormap(pol_2d)
        # Far pixels have pol_2d ≈ 0.0 (depth = depth_max → normalised to 0).
        # COLORMAP_JET maps 0 → blue, not black.  Explicitly set them to black so
        # the policy window matches the raw window.  This is display-only — the policy
        # receives 0.0 for these pixels regardless of how they are coloured here.
        pol_bgr[pol_2d < (1.0 / 255.0)] = (0, 0, 0)
        # Upscale to full camera width with nearest-neighbour so individual policy
        # pixels are clearly visible without bilinear blurring.
        disp_h = max(1, round(cam_w * mushr_env._policy_depth_h / mushr_env._policy_depth_w))
        pol_bgr = cv2.resize(pol_bgr, (cam_w, disp_h), interpolation=cv2.INTER_NEAREST)
        # Active noise settings overlay at the bottom of each policy window.
        noise_txt = (
            f"g={env_cfg.depth_noise_sigma_m:.3f} "
            f"inv={env_cfg.depth_invalid_pixel_prob:.2f} "
            f"col={env_cfg.depth_color_noise_prob:.2f} "
            f"edge={env_cfg.depth_edge_blur_sigma_m:.3f}@{env_cfg.depth_edge_gradient_threshold_m:.2f}"
        )
        cv2.putText(
            pol_bgr, noise_txt,
            (2, disp_h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA,
        )
        _write_atomic(os.path.join(out_dir, f"policy_{suffix}.png"), pol_bgr)'''

new_viz_write = '''    def _write_camera(cam_obj, suffix: str) -> None:
        """Write raw RGB and policy binary mask frames to disk."""
        rgb_tensor = cam_obj.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
        raw_bgr = cv2.cvtColor(rgb_tensor, cv2.COLOR_RGB2BGR)
        _write_atomic(os.path.join(out_dir, f"raw_{suffix}.png"), raw_bgr)

        # ── Policy vision ──────────────────────────────────
        pol_flat = mushr_env._process_single_rgb(cam_obj)
        pol_2d   = pol_flat[0].reshape(
            mushr_env._policy_depth_h, mushr_env._policy_depth_w,
        ).cpu().numpy()
        
        pol_bgr = cv2.cvtColor((pol_2d * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        disp_h = max(1, round(cam_w * mushr_env._policy_depth_h / mushr_env._policy_depth_w))
        pol_bgr = cv2.resize(pol_bgr, (cam_w, disp_h), interpolation=cv2.INTER_NEAREST)
        _write_atomic(os.path.join(out_dir, f"policy_{suffix}.png"), pol_bgr)'''

viz_content = viz_content.replace(old_viz_write, new_viz_write)

old_viz_loop = '''        while simulation_app.is_running():
            throttle, steer = key_state.get_actions()
            actions = torch.tensor([[throttle, steer]], device=device)
            env.step(actions)
            _write_camera(mushr_env.camera_left,  "left")
            _write_camera(mushr_env.camera_right, "right")'''

new_viz_loop = '''        while simulation_app.is_running():
            throttle, steer = key_state.get_actions()
            actions = torch.tensor([[throttle, steer]], device=device)
            env.step(actions)
            _write_camera(mushr_env.camera_center,  "center")'''

viz_content = viz_content.replace(old_viz_loop, new_viz_loop)

with open(viz_path, "w") as f:
    f.write(viz_content)

print("Modification done for play and visualize!")
