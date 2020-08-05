from setuptools import setup, find_packages
import pathlib


setup(
    name='pyimgaug3d',
    version='0.3',
    description='Python image augmentation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='',
    author='Siyu Liu',
    author_email='siyu.liu2@uqconnect.edu.au',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires='>=3.7, <4',
    install_requires=['numpy<1.19.0', 'tensorflow>2.0.0', 'tensorflow-addons', 'nibabel']

)