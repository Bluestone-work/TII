from setuptools import find_packages, setup

package_name = 'gnn_marl_training'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wj',
    maintainer_email='wj@todo.todo',
    description='GNN-based Multi-Agent Reinforcement Learning Training',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'train_gnn_mappo = gnn_marl_training.train_gnn_mappo:main',
            'train_gnn_mappo_full = gnn_marl_training.train_gnn_mappo_full:main',
            'test_gnn_mappo = gnn_marl_training.test_gnn_mappo:main',
            # TurtleBot3 实机部署节点（每台机器人独立运行）
            'robot_policy_node = gnn_marl_training.robot_policy_node:main',
        ],
    },
)
