import setuptools

with open("README.md", "r") as fh:
    long_desc = fh.read()

setuptools.setup(
    name="pygeo3d",
    version="0.0.8",
    author="Divyajeet Singh",
    author_email="knightt1821@gmail.com",
    description="provides a 3D co-ordinate system using Vectors in Python",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    url="https://github.com/divyajeettt/pygeo3d",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "matplotlib",
        "numpy",
        "pyvectors",
        "linear_equations"
    ],
)