from setuptools import setup, find_packages
import os

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='btgym',
    version='0.1.0',
    packages=['btgym'],
    install_requires=required,
    author='HPCL-EI',
    author_email='hpcl_ei@163.com',
    description='A Platform for Behavior Tree Generation and Evaluation',
    url='https://github.com/HPCL-EI/BTGym',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)

