"""
Modal runner for AASR Experiment E1.

This file does not perform the balancing itself.
It launches audit_train_distribution.py inside Modal.

The Modal container:
- installs requests
- gets access to the private Hugging Face dataset
- mounts the shared asar-artifacts volume
- saves E1 outputs permanently in the volume

No GPU is required because this step only processes data.
"""

import modal


app = modal.App("aasr-e1-balance-data")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests>=2.32")
    .add_local_file(
        "experiments/E1_balanced_data/audit_train_distribution.py",
        remote_path="/workspace/audit_train_distribution.py",
    )
)


artifacts = modal.Volume.from_name(
    "asar-artifacts",
    create_if_missing=True,
)


@app.function(
    image=image,
    timeout=60 * 30,
    volumes={
        "/workspace/artifacts": artifacts,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def run_balance():
    """
    Execute the E1 balancing script inside Modal.

    The generated datasets and audit files are saved under:
    /workspace/artifacts/experiments/E1_balanced_data
    """

    import os
    import subprocess

    os.environ["E1_OUTPUT_DIR"] = (
        "/workspace/artifacts/"
        "experiments/E1_balanced_data"
    )

    subprocess.run(
        [
            "python",
            "-u",
            "/workspace/audit_train_distribution.py",
        ],
        check=True,
        env=os.environ.copy(),
    )

    artifacts.commit()


@app.local_entrypoint()
def main():
    """Run the E1 balancing job on Modal."""

    run_balance.remote()