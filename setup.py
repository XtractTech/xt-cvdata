import setuptools, os

PACKAGE_NAME = 'xt-cvdata'
VERSION = '0.5.1'
AUTHOR = 'Xtract AI'
EMAIL = 'info@xtract.ai'
DESCRIPTION = 'Utilities for building and working with computer vision datasets'
GITHUB_URL = 'https://github.com/XtractTech/xt-cvdata'

parent_dir = os.path.dirname(os.path.realpath(__file__))

with open(f'{parent_dir}/README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=GITHUB_URL,
    packages=[
        'xt_cvdata',
        'xt_cvdata.apis',
        'xt_cvdata.datasets',
        'xt_cvdata.transforms'
    ],
    provides=['xt_cvdata'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'torchvision',
        'h5py',
        'pillow',
        'matplotlib',
        'tqdm',
        'pycocotools'
    ],
)
