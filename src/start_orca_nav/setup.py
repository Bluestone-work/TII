from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'start_orca_nav'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Multi-robot navigation using Nav2 with ORCA collision avoidance',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'orca_nav_node = start_orca_nav.orca_nav_node:main',
            'orca_nav_node_nav2 = start_orca_nav.orca_nav_node_nav2:main',
        ],
    },
)
