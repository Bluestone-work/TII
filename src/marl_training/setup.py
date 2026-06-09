from setuptools import setup
import os
from glob import glob

package_name = 'marl_training'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='MARL MAPPO training',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train_mapppo = marl_training.train_mappo:main',
            'test_mappo = marl_training.test_mappo:main',
        ],
    },
)
    