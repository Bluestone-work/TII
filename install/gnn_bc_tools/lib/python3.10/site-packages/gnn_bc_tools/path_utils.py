import os
import sys
import importlib.util
import subprocess
from pathlib import Path
from typing import List, Tuple


def inject_workspace_paths() -> Tuple[Path, str, str]:
    """Ensure src/build packages are importable for this workspace and Ray workers."""
    repo_root = Path(__file__).resolve().parents[3]

    candidate_paths = [
        repo_root / "src" / "gnn_marl_training",
        repo_root / "src" / "start_orca_nav",
        repo_root / "src" / "gnn_bc_tools",
        repo_root / "build" / "gnn_marl_training",
        repo_root / "build" / "start_orca_nav",
        repo_root / "build" / "gnn_bc_tools",
        repo_root / "build" / "gnn_marl_training" / "build" / "lib",
        repo_root / "build" / "start_orca_nav" / "build" / "lib",
        repo_root / "build" / "gnn_bc_tools" / "build" / "lib",
    ]
    ordered_existing_paths = [str(path) for path in candidate_paths if path.exists()]

    for p in ordered_existing_paths:
        while p in sys.path:
            sys.path.remove(p)
    for p in reversed(ordered_existing_paths):
        sys.path.insert(0, p)

    py_entries = ordered_existing_paths + [
        p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p
    ]
    py_dedup = []
    for p in py_entries:
        if p not in py_dedup:
            py_dedup.append(p)
    os.environ["PYTHONPATH"] = os.pathsep.join(py_dedup)

    install_dir = repo_root / "install"
    if install_dir.exists():
        ws_prefixes = [
            str(p) for p in install_dir.iterdir()
            if p.is_dir() and not p.name.startswith("_") and p.name != "COLCON_IGNORE"
        ]
        existing_ament = [
            p for p in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if p
        ]
        ament_entries = ws_prefixes + [p for p in existing_ament if p not in ws_prefixes]
        os.environ["AMENT_PREFIX_PATH"] = os.pathsep.join(ament_entries)

    return repo_root, os.environ["PYTHONPATH"], os.environ.get("AMENT_PREFIX_PATH", "")


def _missing_modules(required_modules: List[str]) -> List[str]:
    return [m for m in required_modules if importlib.util.find_spec(m) is None]


def ensure_runtime_modules(required_modules: List[str], runner_module: str) -> None:
    """
    Ensure required modules are available.
    If missing, try to re-exec with ros2 conda python automatically.
    """
    missing = _missing_modules(required_modules)
    if not missing:
        return

    current_py = Path(sys.executable).resolve()
    candidates: List[Path] = []

    override_py = os.environ.get("GNN_BC_PYTHON")
    if override_py:
        candidates.append(Path(override_py).expanduser())

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "bin" / "python")

    candidates.append(Path.home() / "anaconda3" / "envs" / "ros2" / "bin" / "python")

    dedup_candidates: List[Path] = []
    for c in candidates:
        try:
            rc = c.resolve()
        except FileNotFoundError:
            rc = c
        if rc not in dedup_candidates:
            dedup_candidates.append(rc)

    probe_code = (
        "import importlib.util,sys;"
        "mods=sys.argv[1:];"
        "missing=[m for m in mods if importlib.util.find_spec(m) is None];"
        "sys.exit(0 if not missing else 1)"
    )

    for py_bin in dedup_candidates:
        if not py_bin.exists():
            continue
        if py_bin.resolve() == current_py:
            continue

        probe = subprocess.run(
            [str(py_bin), "-c", probe_code, *required_modules],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode != 0:
            continue

        if os.environ.get("GNN_BC_REEXECED") == "1":
            break

        env = os.environ.copy()
        env["GNN_BC_REEXECED"] = "1"
        os.execve(
            str(py_bin),
            [str(py_bin), "-m", runner_module, *sys.argv[1:]],
            env,
        )

    missing_str = ", ".join(missing)
    raise RuntimeError(
        "Missing Python modules: "
        f"{missing_str}\n"
        f"Current Python: {current_py}\n"
        "Please activate the ros2 conda env and install dependencies, e.g.:\n"
        "  source /home/wj/anaconda3/etc/profile.d/conda.sh\n"
        "  conda activate ros2\n"
        "  pip install gymnasium 'ray[rllib]' torch numpy\n"
        "Then source ROS/workspace and retry:\n"
        "  source /opt/ros/humble/setup.bash\n"
        "  source /home/wj/work/multi-robot-exploration-rl/install/setup.bash"
    )
