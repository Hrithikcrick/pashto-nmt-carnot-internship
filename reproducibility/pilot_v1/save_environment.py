import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


packages = [
    "torch",
    "transformers",
    "peft",
    "datasets",
    "pandas",
    "numpy",
    "sacrebleu",
    "unbabel-comet",
    "pytorch-lightning",
    "torchmetrics",
]

package_versions = {}

for package in packages:
    try:
        package_versions[package] = version(package)
    except PackageNotFoundError:
        package_versions[package] = "not installed"

information = {
    "snapshot_name": "pilot-study-v1",
    "python_version": sys.version,
    "python_executable": sys.executable,
    "operating_system": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "package_versions": package_versions,
}

output = Path("reproducibility/pilot_v1/runtime_environment.json")

output.write_text(
    json.dumps(
        information,
        indent=4,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print(f"Runtime information saved to: {output}")
