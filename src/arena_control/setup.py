import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'arena_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'),
        glob(os.path.join('launch', '*.launch.py'))),
],
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Thomas Byrne',
    maintainer_email='tbyrne217@gmail.com',
    description='ROS 2 control package for the AI Fight Arena reinforcement-learning simulation.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
   'console_scripts': [
    'fighter_alpha = arena_control.fighter_alpha:main',
    'fighter_bravo = arena_control.fighter_bravo:main',
    'fight_controller = arena_control.fight_controller:main',
'bravo_ai = arena_control.bravo_ai:main',
],
},
)
