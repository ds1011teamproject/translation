import os
from setuptools import setup, find_packages, Command


class CleanCommand(Command):
    """ cleans the build directory """
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="nlpt",
    version="0.1.0",
    author="nlpteam",
    description="base code base for the nlp project",
    packages=find_packages(),  # finds all directories and sub directories with __init__.py
    long_description=read('changelog.md'),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: Public Domain",
        "Programming Language :: Python :: 3.6",
    ],  # see list at https://pypi.org/pypi?%3Aaction=list_classifiers
    install_requires=[
        'numpy',
        'pandas',
        'torch>=0.4',
        'torchvision>=0.2',
        'tqdm'
    ],
    python_requires='==3.6.*',
    zip_safe=False,
    cmdclass={'clean': CleanCommand}
)
