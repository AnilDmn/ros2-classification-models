from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ml_classifiers_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # CSV dosyalarının install klasörüne kopyalanmasını sağlar
        (os.path.join('share', package_name, 'data'), glob('data/*.csv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anild',
    maintainer_email='ad.anilduman@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fruit_node = ml_classifiers_pkg.fruit_node:main',
            'iris_node = ml_classifiers_pkg.iris_node:main',
            'cancer_node = ml_classifiers_pkg.cancer_node:main',
            'penguin_node = ml_classifiers_pkg.penguin_node:main',
        ],
    },
)

