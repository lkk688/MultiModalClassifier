from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy", "tensorflow", "pillow", "matplotlib"]

setup(
    name="MultimodalClassifier",
    version="0.0.1",
    author="Kaikai Liu",
    author_email="kaikai.liu@sjsu.edu",
    description="A multimodal classifier based on Tensorflow and Pytorch",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/lkk688/MultiModalClassifier",
    packages=['TFClassifier'], #find_packages(exclude=['DatasetTools', 'tests', 'data', 'outputs']),
    install_requires=['numpy'],
    classifiers=[
        "Programming Language :: Python :: 3.8",
    ],
)
