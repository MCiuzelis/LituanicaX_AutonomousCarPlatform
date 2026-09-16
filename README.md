# LituanicaX — Intelligent Robot Competition (CAD)

Mechanical and electrical design files for the LituanicaX autonomous RC car:
a Raspberry Pi 5–based robot that navigates a cone-track circuit using a
depth camera.

This branch of the repository contains **only the CAD**. The RL training
pipeline (Isaac Lab) and the Raspberry Pi deployment code were removed from
the tree but are fully preserved in git history — check out any commit
before the "CAD-only" commit to get them back.

---

## Repository Structure

```
CAD/
├── PCB/                        # Main driver board (KiCad)
│   ├── Main_V2/                # KiCad project — schematic + layout, V2
│   │   ├── Main.kicad_pro
│   │   ├── Main.kicad_sch
│   │   ├── Main.kicad_pcb
│   │   └── jlcpcb/             # Fabrication outputs for JLCPCB
│   │       ├── gerber/         # Gerber + drill files
│   │       └── production_files/   # BOM, CPL (pick-and-place), gerber zip
│   ├── RC_CAR.kicad_sym        # Project symbol library
│   ├── RC_CAR.pretty/          # Project footprint library
│   ├── Downloaded_IC_Models/   # Vendor symbols / footprints / 3D models
│   └── board_V2.step           # 3D export of the assembled board
│
├── Solidworks/                 # Mechanical design (SolidWorks)
│   ├── !Main.SLDASM            # Top-level assembly
│   ├── Case.SLDPRT             # Electronics enclosure
│   ├── CameraMount.SLDPRT      # Depth-camera mount + adapter
│   ├── CameraAdapter.SLDPRT
│   ├── MountAdapter.SLDPRT     # Chassis-to-case adapter
│   ├── servoMount.SLDPRT       # Steering-servo bracket
│   ├── ExtendedHeaders.SLDPRT
│   ├── board_V2.SLDPRT         # PCB reference geometry
│   ├── Raspberry_pi_5.SLDPRT   # Off-the-shelf part models
│   ├── RP5_ActiveCooler.SLDPRT
│   ├── RPiAIHaloHat.SLDPRT
│   ├── SparkfunVoltmeter.SLDPRT
│   ├── STEP/                   # STEP versions of the reference parts
│   └── export/                 # Print-ready parts (3MF exports are not tracked)
│
└── Timer/                      # Lap-timer board (KiCad, ESP32 + ToF sensor)
    ├── Kicad/                  # Timer.kicad_pro / _sch / _pcb
    ├── Library/                # TimerLib symbol + footprint library
    └── DownloadedICs/          # Vendor symbols / footprints / 3D models
```

---

## Hardware Overview

| Component | Details |
|---|---|
| Robot | Ackermann-steered RC car chassis, custom 3D-printed mounts and case |
| Onboard compute | Raspberry Pi 5 |
| Depth sensor | ArduCam ToF camera on the custom `CameraMount` |
| Main PCB (`CAD/PCB`) | Motor driver, servo control, power management, Pi header breakout |
| Timer PCB (`CAD/Timer`) | ESP32-based lap timer with TMF8801 ToF sensor, USB-C, RS-485 |

---

## Tooling

| Files | Tool |
|---|---|
| `*.kicad_pro`, `*.kicad_sch`, `*.kicad_pcb` | KiCad 8+ |
| `*.SLDPRT`, `*.SLDASM` | SolidWorks 2024+ |
| `*.step` / `*.stp` | Any CAD viewer |

### Regenerating fabrication files

Open `CAD/PCB/Main_V2/Main.kicad_pro` in KiCad and run the
*Fabrication Toolkit* / JLCPCB plugin — outputs land in `Main_V2/jlcpcb/`.

### 3D-printed parts

Open `CAD/Solidworks/!Main.SLDASM` and export the printed parts
(`Case`, `CameraMount`, `CameraAdapter`, `MountAdapter`, `servoMount`) to
3MF/STL for your slicer. Slicer exports (`*.3mf`) are ignored by git —
the SolidWorks sources are the single source of truth.

---

## What is *not* tracked

`.gitignore` excludes KiCad auto-backups (`*-backups/`), VS Code local
history (`.history/`), KiCad local project settings (`*.kicad_prl`),
`*.bak` library backups, SolidWorks lock files (`~$*`), 3MF/G-code slicer
exports, and OS noise.
