from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> List[str]:
    """
    This function reads a requirements file and returns a list of packages.
    It removes any empty lines and comments.
    """
    requirements = []
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
    
    requirements = [req.replace('\n', '') for req in requirements]

    if 'HYPEN_E_DOT' in requirements:
        requirements.remove('HYPHEN_E_DOT')

setup(
    name = 'ML-Project',
    version = '0.1.0',
    author = 'Dhanush R M',
    author_email = 'techymode6115@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)