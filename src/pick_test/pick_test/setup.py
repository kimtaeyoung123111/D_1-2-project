from setuptools import find_packages, setup

package_name = 'pick_test'

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
    maintainer='kim',
    maintainer_email='kim@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'robot_move_wh = pick_test.robot_move_wh:main',
        'detection_opencv = pick_test.detection_opencv:main',
        'belt_control_node = pick_test.belt_control_node:main'
        ],
    },
)
