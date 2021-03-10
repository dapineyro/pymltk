import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pymltk",
    version="0.0.5",
    author="David Piñeyro",
    author_email="dapineyro.dev@gmail.com",
    license="GPL-3",
    description="Python Machine Learning Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dapineyro/pymltk",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.8',
    # If any package contains *.r files, include them:
    package_data={'': ['*.r', '*.R']},
    include_package_data=True
)
