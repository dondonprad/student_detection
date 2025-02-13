from setuptools import find_packages, setup


setup(
    name = 'studet_detection',
    version = '0.0.1',
    author = 'Doni Pradana',
    author_email= 'donipradana29@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)