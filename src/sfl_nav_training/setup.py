from setuptools import find_packages, setup

package_name = "sfl_nav_training"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "gymnasium",
    ],
    zip_safe=True,
    maintainer="wj",
    maintainer_email="wj@todo.todo",
    description="Sampling-for-learnability style IPPO training package over Gazebo+RViz multi-robot nav.",
    license="Apache-2.0",
    extras_require={
        "train": ["ray[rllib]", "torch"],
        "test": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "train_sfl_ippo = sfl_nav_training.train_sfl_ippo:main",
        ],
    },
)
