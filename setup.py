import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="nlptranslate",
    version="0.1.0",
    author="nlpteam",
    description="base code base for the nlp project",
    packages=find_packages(),
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
    ],
    install_requires=[
        'numpy',
        'pandas',
        'torch==0.4',
        'torchvision==0.2',
        'tqdm'
    ],
    python_requires='>=3',
    zip_safe=False
)
