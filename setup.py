import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DeepSR",
    version="1.2.0",
    author="Hakan Temiz, Hasan Sakir Bilge",
    author_email="htemiz@artvin.edu.tr, bilge@gazi.edu.tr",
    description="A framework for the task of Super Resolution with Deep Learning Algorithms based on Keras framework.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/htemiz/DeepSR",
    packages=setuptools.find_packages(),
    keywords="super resolution deep learning",
    # python_requires='>=3',
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst', '*.md', 'samples/*.*'],
    },
    exclude_package_data={'': ['']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    scripts=[],
    install_requires=[],
)