import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tf_osqp",
    version="0.1.0",
    author="Rishabh Jha",
    author_email="rishabhjha.code@gmail.com",
    maintainer="Rishabh Jha",
    maintainer_email="rishabhjha.code@gmail.com",
    description="Tensorflow implementation of OSQP solver (unofficial)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rishabhjha33/tf-osqp",
    packages=setuptools.find_packages(),
    license="Apache 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
	"Development Status :: 3 - Alpha",
	"License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
	"Intended Audience :: Science/Research",
	"Natural Language :: English",
    ],
    python_requires='>=3.5',
    install_requires = [
        'numpy >= 1.18',
        'tensorflow >= 2.4rc3'
        ]
)
