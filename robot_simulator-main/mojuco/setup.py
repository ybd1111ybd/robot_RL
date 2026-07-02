from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'mujoco_simulator'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/mujoco_simulator', ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('*.mjcf.xml')),
        (os.path.join('share', package_name, 'models', 'meshes'), glob('meshes/*.STL')),
    ],
    install_requires=['setuptools', 'mujoco', 'numpy'],
    zip_safe=True,
    maintainer='Developer',
    maintainer_email='user@example.com',
    description='MuJoCo physics simulation for dual-arm robot with ROS2 interface',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mujoco_sim_node = mujoco_simulator.mujoco_sim_node:main',
        ],
    },
)
