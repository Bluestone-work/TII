from setuptools import find_packages, setup

package_name = 'gnn_bc_tools'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wj',
    maintainer_email='wj@todo.todo',
    description='BC data collection and pretraining tools for GNN-MAPPO using ORCA/DWA experts.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'collect_orca_dwa_bc = gnn_bc_tools.collect_orca_dwa_bc:main',
            'pretrain_mappo_bc = gnn_bc_tools.pretrain_mappo_bc:main',
            'run_orca_dwa_bc_pipeline = gnn_bc_tools.run_orca_dwa_bc_pipeline:main',
            'tune_orca_dwa_map1 = gnn_bc_tools.tune_orca_dwa_map1:main',
        ],
    },
)
