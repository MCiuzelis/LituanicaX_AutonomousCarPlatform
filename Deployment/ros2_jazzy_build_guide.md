# Building ROS 2 Jazzy from Source on Raspberry Pi 5 (Debian Trixie)

Step-by-step guide for building ROS 2 Jazzy on **Raspberry Pi 5** running **Debian 13 "Trixie"** (Pi OS Lite).

---

## Prerequisites & System Prep

Before pulling down the code, update the system and install foundational build tools.

```bash
# Bring the system package index up to date
sudo apt update && sudo apt upgrade -y

# Install native compilation tools and modern PyQt/SIP engines available in Trixie
sudo apt install -y build-essential cmake git python3-pip python3-pytest-cov \
wget python3-setuptools python3-flake8-docstrings libacl1-dev libssl-dev \
libasio-dev libtinyxml2-dev pyqt5-dev-tools sip-tools
```

## Install ROS Python Bootstrap Tools

Debian Trixie does not include ROS bootstrap infrastructure natively in apt, so install via pip.

```bash
sudo pip3 install --break-system-packages -U \
  colcon-common-extensions \
  rosdep \
  vcstool \
  catkin_pkg \
  rosdistro
```

---

## 1. Initialize Rosdep

```bash
sudo rosdep init
rosdep update
```

## 2. Fetch the ROS 2 Source Code

```bash
mkdir -p ~/ros2_jazzy/src
cd ~/ros2_jazzy
vcs import src --input https://raw.githubusercontent.com/ros2/ros2/jazzy/ros2.repos
```

## 3. Resolve Dependencies (Trixie Workarounds)

Debian Trixie has deprecated or renamed some packages. The `--skip-keys` flags below must each be specified **separately** (space-separated values in one quoted string are parsed as a single key name and will not work).

```bash
rosdep install --from-paths src --ignore-src --rosdistro jazzy -y \
  --skip-keys python3-sip \
  --skip-keys python3-sip-dev \
  --skip-keys python3-vcstool \
  --skip-keys python3-rosdistro-modules \
  --skip-keys python3-catkin-pkg-modules \
  --skip-keys fastcdr \
  --skip-keys rti-connext-dds-6.0.1 \
  --skip-keys urdfdom_headers
```

### Additional system packages needed on Trixie

`python3-sip-dev` does not exist on Debian Trixie. Install its replacements plus other missing dependencies:

```bash
sudo apt install -y \
  liborocos-kdl-dev lttng-tools clang-format uncrustify libsqlite3-dev \
  libbullet-dev python3-cairo python3-babeltrace libxml2-utils pydocstyle \
  python3-pyqt5.sip python3-sipbuild sip-tools libqt5svg5-dev
```

## 4. Compile the Workspace

```bash
cd ~/ros2_jazzy
colcon build --symlink-install --continue-on-error \
  --parallel-workers 2 \
  --cmake-args -DBUILD_TESTING=OFF
```

**Flags explained:**
- `--symlink-install` — symlink Python packages for faster iteration
- `--continue-on-error` — continue if a package fails
- `--parallel-workers 2` — limits parallel builds to avoid OOM on 4 GB Pi 5
- `-DBUILD_TESTING=OFF` — skips tests (~50% faster build)

**Expected build time:** ~2 hours on Pi 5 (4 cores, 4 GB RAM).

## 5. Environment Activation

### One-time

```bash
source ~/ros2_jazzy/install/local_setup.bash
```

### Persistent (auto-load on terminal start)

```bash
echo "source ~/ros2_jazzy/install/local_setup.bash" >> ~/.bashrc
```

Verify with:

```bash
ros2 -h
```

## 5.1. Build the LituanicaX Deployment Workspace

After the ROS 2 underlay works, rebuild the robot workspace from this repo:

```bash
cd ~/LituanicaX_InteligentRobotCompetition
Deployment/install.sh
```

The script expects the ROS 2 source build at `~/ros2_jazzy`. If you built it
elsewhere, point the installer at that underlay:

```bash
ROS_UNDERLAY=/path/to/ros2_jazzy Deployment/install.sh
```

To run the car stack:

```bash
source ~/ros2_jazzy/install/setup.bash
source Deployment/Ros2/install/setup.bash
ros2 launch lituanicax_rc_car ros_autopilot.launch.py
```

## 6. Verify with Talker/Listener Demo

**Terminal 1:**
```bash
source ~/ros2_jazzy/install/setup.bash
ros2 run demo_nodes_cpp talker
```

**Terminal 2:**
```bash
source ~/ros2_jazzy/install/setup.bash
ros2 run demo_nodes_cpp listener
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `rosdep` fails "Unable to locate package" | Add the missing key to `--skip-keys` (as a separate flag) and install the equivalent package manually |
| `fatal error: sys/acl.h` | `sudo apt install -y libacl1-dev` |
| Out of memory during build | Reduce to `--parallel-workers 1` |
| `rosdep` skip-keys not working | Each key needs its own `--skip-keys` flag — do **not** use `--skip-keys "a b c"` |
