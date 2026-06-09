from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'start_reinforcement_learning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
         glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'maps'),
         glob(os.path.join('maps', '*'))),
        (os.path.join('share', package_name, 'config'),
         glob(os.path.join('config', '*.yaml')) + glob(os.path.join('config', '*.rviz'))),
        (os.path.join('share', package_name, 'scripts'),
         glob(os.path.join('scripts', '*.sh'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='unruly',
    maintainer_email='unruly@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'run_matd3 = start_reinforcement_learning.matd3_main:main',
            'run_maddpg = start_reinforcement_learning.matd3_main:main',  # 保留旧名称作为别名
        ],
    },
)
