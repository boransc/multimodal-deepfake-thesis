from pathlib import Path
import json
import shutil
from datetime import datetime

def setup_experiment(
    project_root,
    experiment_id,
    experiment_name,
    config,
    notebook_path=None,
):
    project_root = Path(project_root)

    run_name = f"{experiment_id}_{experiment_name}"
    run_dir = project_root / "experiments" / run_name

    dirs = {
        "run": run_dir,
        "manifests": run_dir / "manifests",
        "results": run_dir / "results",
        "checkpoints": run_dir / "checkpoints",
        "plots": run_dir / "plots",
        "notebook": run_dir / "notebook",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    config = dict(config)
    config["experiment_id"] = experiment_id
    config["experiment_name"] = experiment_name
    config["run_name"] = run_name
    config["created_at"] = datetime.now().isoformat(timespec="seconds")

    config_path = run_dir / "config.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    if notebook_path is not None:
        notebook_path = Path(notebook_path)

        if notebook_path.exists():
            snapshot_path = dirs["notebook"] / f"{run_name}_notebook_snapshot.ipynb"
            shutil.copy2(notebook_path, snapshot_path)
            print("Notebook snapshot saved:", snapshot_path)
        else:
            print("Notebook path not found:", notebook_path)

    print("Experiment created:", run_dir)

    return run_name, dirs