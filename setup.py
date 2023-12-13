from setuptools import setup, find_packages

setup(
    name='LifelongNeuro',
    version='1.0',
    description='Lifelong learning with neuro datasets.',
    url='https://github.com/camgbus/LifelongNeuro',
    keywords='python setuptools',
    packages=find_packages(include=['lln', 'lln.*']),
)