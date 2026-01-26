# nvidia-driver-reload

Reload NVIDIA drivers **without rebooting** on headless Linux GPU servers.

## Why?

Driver updates, version mismatches, and stuck GPU states typically require a full server reboot. On production GPU servers, that means:

- Killing running workloads
- Minutes of downtime
- Lost compute time / rental revenue

This tool gracefully stops GPU workloads, unloads/reloads kernel modules, and restarts everything — no reboot required.

## Quick Start

```bash
# Check current status
sudo python3 nvidia_driver_reload.py --status

# Dry run (see what would happen)
sudo python3 nvidia_driver_reload.py --reload --dry-run

# Full driver reload
sudo python3 nvidia_driver_reload.py --reload --yes

# GPU reset only (lighter, faster)
sudo python3 nvidia_driver_reload.py --reset
```

## Features

- **Hot-reload drivers** — Unload and reload kernel modules in the correct order
- **Intelligent process detection** — Uses NVML API and `fuser` to find *actual* GPU users, not static process lists
- **Container-aware** — Gracefully stops Docker/Podman GPU containers, restarts daemons, restarts containers
- **Install new drivers** — Install a `.run` driver file and reload without reboot
- **Smart error detection** — Detects when reboot is actually required (XID 79, hardware failures)
- **Handles nvidia_drm.modeset=1** — Unbinds VT consoles and framebuffer automatically
- **Rollback support** — Saves state before operations, can rollback on failure
- **Safe by design** — Never kills system processes, uses authoritative detection only

## Installation

```bash
# Just the script (no dependencies required)
curl -O https://raw.githubusercontent.com/YOUR_USERNAME/nvidia-driver-reload/main/nvidia_driver_reload.py
chmod +x nvidia_driver_reload.py

# Optional: better GPU detection and container control
pip install nvidia-ml-py docker psutil
```

## Usage

```bash
# Show GPU status and reload feasibility
sudo python3 nvidia_driver_reload.py --status

# Comprehensive health check
sudo python3 nvidia_driver_reload.py --verify

# Full driver reload (stops workloads, unloads modules, reloads, restarts)
sudo python3 nvidia_driver_reload.py --reload

# Install new driver and reload
sudo python3 nvidia_driver_reload.py --reload --driver /path/to/NVIDIA-Linux-x86_64-550.127.08.run

# GPU reset only (faster, doesn't unload modules)
sudo python3 nvidia_driver_reload.py --reset

# Stop all GPU workloads (manual control)
sudo python3 nvidia_driver_reload.py --stop

# Restart previously stopped workloads
sudo python3 nvidia_driver_reload.py --start

# Rollback from failed reload
sudo python3 nvidia_driver_reload.py --rollback
```

### Options

| Flag | Description |
|------|-------------|
| `--status`, `-s` | Show current NVIDIA status |
| `--verify` | Comprehensive nvidia-smi health check |
| `--reload`, `-r` | Full driver reload |
| `--reset` | GPU reset only (nvidia-smi --gpu-reset) |
| `--stop` | Stop all GPU workloads |
| `--start` | Restart stopped workloads |
| `--rollback` | Rollback from failed state |
| `--driver PATH` | Install driver .run file before reload |
| `--yes`, `-y` | Skip confirmation prompts |
| `--dry-run` | Show what would happen |
| `--verbose`, `-v` | Verbose output |
| `--json` | JSON output for status/verify |

## Requirements

- Linux (tested on Ubuntu 22.04/24.04, Debian 12)
- Root privileges
- Python 3.8+
- **Headless server** — No X11/Wayland display server using the GPU

### Optional Dependencies

```bash
pip install nvidia-ml-py   # Better GPU detection via NVML
pip install docker         # Docker container management
pip install psutil         # Process detection
```

## How It Works

1. **Detect GPU state** — Check for fatal errors, running processes, module usage
2. **Stop GPU containers** — Gracefully stop Docker/Podman containers using GPU
3. **Stop services** — nvidia-persistenced, nvidia-fabricmanager
4. **Kill GPU processes** — Only processes detected by NVML/fuser (not by name)
5. **Restart systemd-logind** — Releases DRM handles (the #1 hidden culprit)
6. **Handle modeset** — Unbind VT consoles and framebuffer if nvidia_drm.modeset=1
7. **Unload modules** — nvidia_drm → nvidia_modeset → nvidia_uvm → nvidia
8. **Install driver** — (optional) Run the .run installer
9. **Load modules** — nvidia → nvidia_uvm → nvidia_modeset → nvidia_drm
10. **Rebind console** — Restore framebuffer and VT consoles
11. **Verify driver** — Comprehensive nvidia-smi health check
12. **Restart Docker** — Required to refresh nvidia-container-toolkit paths
13. **Restart containers** — Bring back GPU workloads

## Intelligent Process Detection

This tool does **NOT** use static process name lists (which kill innocent `python`, `docker`, `containerd` processes). Instead, it uses authoritative detection:

| Method | What it catches |
|--------|-----------------|
| **NVML API** | All processes actively using GPU compute/graphics resources |
| **fuser /dev/nvidia*** | All processes holding open handles to NVIDIA device files |

If a process isn't detected by these tools, it's not using the GPU — leave it alone. This approach works for any GPU workload without manual updates:

- CUDA applications
- PyTorch/TensorFlow training jobs
- Docker containers with `--gpus`
- Podman GPU containers
- QEMU/KVM GPU passthrough
- FFmpeg hardware encoding
- Any other GPU workload

### Safety Guarantees

- Never kills PIDs < 100 (kernel threads, init)
- Never kills critical system processes (systemd, dbus, kworker, etc.)
- Only terminates processes confirmed to be holding GPU resources

## When Reboot Is Required

The script automatically detects scenarios where reload won't work:

| Condition | Why |
|-----------|-----|
| XID 79 | GPU fell off PCIe bus — hardware issue |
| GSP firmware failure | Firmware needs full reset |
| NULL pointer in nvidia module | Kernel corruption |
| Display server running | X11/Wayland holds GPU — stop it first |

## XID Error Handling

| XID | Severity | Action |
|-----|----------|--------|
| 79 | Fatal | Reboot required |
| 48, 74, 95, 119 | Recoverable | GPU reset works |
| 31, 43, 45, 68, 69, 94 | App fault | Just restart application |
| 61, 62, 63, 64, 92 | Info | Monitor, usually fine |

**Key insight:** Many XIDs previously thought "fatal" are actually recoverable. The script uses timestamp filtering to avoid false positives from old dmesg entries.

## Limitations

- **Headless only** — Display servers prevent module unload
- **CUDA state lost** — No checkpoint/restore, running CUDA jobs are killed
- **NVLink systems** — Fabric Manager version must match driver
- **Screen blanks** — Expected during modeset=1 unbind (console comes back)

## Files

| Path | Purpose |
|------|---------|
| `/var/run/nvidia-reload.lock` | Prevents concurrent runs |
| `/var/lib/nvidia-reload/state.json` | Saved state for rollback |
| `/var/log/nvidia-reload.log` | Operation log |

## References

- [NVIDIA Forums: Reset driver without rebooting](https://forums.developer.nvidia.com/t/reset-driver-without-rebooting-on-linux/40625)
- [CUDA Driver Reload Guide](https://zyao.net/linux/2024/09/29/cuda-driver-reload/)
- [nvidia-container-toolkit #169](https://github.com/NVIDIA/nvidia-container-toolkit/issues/169)
- [Arch Wiki: NVIDIA Tips](https://wiki.archlinux.org/title/NVIDIA/Tips_and_tricks)
- [Arch Forums: nvidia_drm unload](https://bbs.archlinux.org/viewtopic.php?id=295484)
- [NVIDIA XID Errors](https://docs.nvidia.com/deploy/xid-errors/)
- [optimus-manager](https://github.com/Askannz/optimus-manager)
- [GPU Passthrough Scripts](https://github.com/QaidVoid/Complete-Single-GPU-Passthrough)

## License

MIT
