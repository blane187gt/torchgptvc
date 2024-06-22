from setuptools import setup, find_packages

setup(
    name="GPTCRE",
    version="0.1.0",
    description="A monophonic pitch tracker based on a deep convolutional neural network, created with ChatGPT.",
    author="Your Name",
    author_email="laynzch@gmail.com",
    url="https://github.com/Nex432/GPTCRE",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "librosa",
        "matplotlib",
        "tensorflow"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
