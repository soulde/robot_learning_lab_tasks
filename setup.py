"""Installation script for the ``robot_learning_lab_tasks`` package."""

from pathlib import Path

import toml
from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
EXTENSION = toml.load(ROOT / "config" / "extension.toml")

setup(
    name="robot_learning_lab_tasks",
    version=EXTENSION["package"]["version"],
    author=EXTENSION["package"]["author"],
    maintainer=EXTENSION["package"]["maintainer"],
    url=EXTENSION["package"]["repository"],
    description=EXTENSION["package"]["description"],
    keywords=EXTENSION["package"]["keywords"],
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy>=1.23"],
    extras_require={
        "isaaclab": [
            "isaaclab",
            "isaaclab_tasks",
            "robot_learning_lab_datasets",
            "robot_learning_lab_zoo[isaaclab]",
        ],
        "mjlab": ["mjlab", "robot_learning_lab_zoo[mjlab]"],
    },
    entry_points={
        "mjlab.tasks": [
            "robot_learning_lab_tasks=robot_learning_lab_tasks.tasks.mjlab"
        ]
    },
    python_requires=">=3.10",
    zip_safe=False,
)
