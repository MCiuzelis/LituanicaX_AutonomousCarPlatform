# Training Environment

This directory contains the Isaac Sim / Isaac Lab training environment for the
LituanicaX intelligent robot competition.

## Fresh setup

From the repository root:

install uv 

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
./Training/install.sh
```

That command will:

1. Initialize and update the `Training/IsaacLab` git submodule.
2. Force the submodule to the Isaac Sim 5.1-compatible Isaac Lab tag `v2.3.0`.
3. Create or update the `Training/.venv` virtual environment with `uv sync`.

If the submodule is already present, you can also run `uv sync` directly inside
`Training/`, but `./Training/install.sh` is the safer fresh-machine command.

## Notes

- The virtual environment lives at `Training/.venv`.
- The local Isaac Lab packages are installed from the submodule in editable mode.
- The local `tasks` package is installed from `Training/source`.
