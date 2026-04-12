"""
MuSHR policy inference runner — no Isaac Sim / Isaac Lab required.

Runs policy.onnx (preferred) or policy.pt (TorchScript fallback) with
random noise as input, printing telemetry to stdout.

Usage:
    python run_policy.py                  # uses policy.onnx by default
    python run_policy.py --backend torch  # uses policy.pt (TorchScript)
    python run_policy.py --steps 500      # run for 500 steps then exit (0 = forever)
"""

import argparse
import os
import time
import statistics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OBS_DIM   = 11310   # input size  (depth image crop + velocity scalars)
ACT_DIM   = 2       # output size (throttle, steer) in [-1, 1]
TARGET_HZ = 30      # policy runs at 30 Hz on the real robot


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    p.add_argument("--steps", type=int, default=0, help="Steps to run (0 = forever)")
    p.add_argument("--report_every", type=int, default=30, help="Print stats every N steps")
    return p.parse_args()


# ── ONNX backend ─────────────────────────────────────────────────────────────

def load_onnx(path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4   # RP5 has 4 cores
    sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"[ONNX] loaded  {path}")
    print(f"       input : {input_name}  shape={sess.get_inputs()[0].shape}  dtype={sess.get_inputs()[0].type}")
    print(f"       output: {output_name} shape={sess.get_outputs()[0].shape} dtype={sess.get_outputs()[0].type}")
    return sess, input_name, output_name


def run_onnx(path, steps, report_every):
    import numpy as np
    sess, input_name, output_name = load_onnx(path)

    latencies = []
    step = 0

    print(f"\n{'step':>6}  {'throttle':>10}  {'steer':>10}  {'step_ms':>9}  {'avg_hz':>8}")
    print("-" * 55)

    while True:
        obs = np.random.randn(1, OBS_DIM).astype(np.float32)

        t0 = time.perf_counter()
        action = sess.run([output_name], {input_name: obs})[0]
        # RSL-RL exports raw actor mean — clip to [-1, 1] as the env wrapper does during training
        action = action.clip(-1.0, 1.0)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        latencies.append(dt_ms)
        step += 1

        if step % report_every == 0:
            avg_hz = 1000.0 / statistics.mean(latencies[-report_every:])
            print(
                f"{step:>6}  {action[0, 0]:>+10.4f}  {action[0, 1]:>+10.4f}"
                f"  {dt_ms:>8.2f}ms  {avg_hz:>7.1f}Hz"
            )

        if steps and step >= steps:
            break

        # Optional: pace to real 30 Hz (comment out to benchmark max throughput)
        # elapsed = time.perf_counter() - t0
        # sleep = max(0.0, 1.0 / TARGET_HZ - elapsed)
        # time.sleep(sleep)

    print_summary(latencies)


# ── TorchScript backend ──────────────────────────────────────────────────────

def load_torch(path):
    import torch
    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    print(f"[Torch] loaded {path}")
    return model


def run_torch(path, steps, report_every):
    import torch
    model = load_torch(path)

    latencies = []
    step = 0

    print(f"\n{'step':>6}  {'throttle':>10}  {'steer':>10}  {'step_ms':>9}  {'avg_hz':>8}")
    print("-" * 55)

    with torch.no_grad():
        while True:
            obs = torch.randn(1, OBS_DIM)

            t0 = time.perf_counter()
            action = model(obs).clamp(-1.0, 1.0)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            latencies.append(dt_ms)
            step += 1

            if step % report_every == 0:
                avg_hz = 1000.0 / statistics.mean(latencies[-report_every:])
                a = action[0]
                print(
                    f"{step:>6}  {a[0].item():>+10.4f}  {a[1].item():>+10.4f}"
                    f"  {dt_ms:>8.2f}ms  {avg_hz:>7.1f}Hz"
                )

            if steps and step >= steps:
                break

    print_summary(latencies)


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary(latencies):
    print()
    print("=" * 55)
    print("  SUMMARY")
    print(f"  steps      : {len(latencies)}")
    print(f"  mean latency: {statistics.mean(latencies):.2f} ms")
    print(f"  min  latency: {min(latencies):.2f} ms")
    print(f"  max  latency: {max(latencies):.2f} ms")
    if len(latencies) > 1:
        print(f"  stdev       : {statistics.stdev(latencies):.2f} ms")
    print(f"  mean freq   : {1000.0 / statistics.mean(latencies):.1f} Hz  (target: {TARGET_HZ} Hz)")
    print("=" * 55)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"MuSHR policy bench — obs_dim={OBS_DIM}  act_dim={ACT_DIM}  backend={args.backend}")
    print(f"steps={args.steps or 'forever'}  report_every={args.report_every}")
    print()

    if args.backend == "onnx":
        run_onnx(
            os.path.join(SCRIPT_DIR, "policy.onnx"),
            args.steps,
            args.report_every,
        )
    else:
        run_torch(
            os.path.join(SCRIPT_DIR, "policy.pt"),
            args.steps,
            args.report_every,
        )


if __name__ == "__main__":
    main()
