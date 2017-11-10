# dynet_tutorial

## Cloning the repository

You can download a copy of this code with the following command:

```
git clone https://github.com/michaelcapizzi/dynet_tutorial.git
```

## Environment Setup

It is recommended that you use `virtual environment`s for all `python` projects.

For more informatio on `conda`, see [here](https://conda.io/docs/intro.html)
For more information on `virtualenv`, see [here](http://docs.python-guide.org/en/latest/dev/virtualenvs/)

This repository is built with `python 3` and has the following requirements:

 - `numpy`
 - `jupyter`
 - `matplotlib`
 
They can be installed by running `pip install -r requirements.txt`

## Installing `dynet`

These instructions can all be found [here](http://dynet.readthedocs.io/en/latest/python.html#)

### System Requirements

Make sure to have the following installed globally on your machine before attempting installation.

#### `Linux`

```
udo apt-get update
sudo apt-get install python-pip build-essential cmake mercurial
```

#### `Mac`

```
xcode-select --install
brew install cmake hg python 
```

### Installation

These instructions have been taken from [here](http://dynet.readthedocs.io/en/latest/python.html#manual-installation)

`dynet` is written in `C++` with a `python` wrapper (there's also a [`scala` wrapper!](https://github.com/clab/dynet/tree/master/contrib/swig)).  
As a result, installing `dynet` requires three steps:

 1. getting system requirements
 2. compiling the `C++` code
 3. installing the `python` wrapper
 
Anytime you re-install or update `dynet`, you will be required to:

  1. `git pull` from the `dynet` directory
  2. compile the `C++` code (step 2 above)
  3. reinstall the `python` wrapper (step 3 above)

#### Getting system requirements

1. Install `cython`

```
pip install cython
```

#### Compiling the `C++` code

1. Clone the `dynet` repository

```
mkdir dynet-base
cd dynet-base
git clone https://github.com/clab/dynet.git
```

2. Clone the `eigen` repository (requirement for matrix operations)

```
hg clone https://bitbucket.org/eigen/eigen -r 346ecdb
```

3. Compile `C++` (this can take up to 10 minutes)

```
cd dynet
mkdir build
cd build
cmake .. -DEIGEN3_INCLUDE_DIR=../../eigen -DPYTHON=`which python`
make -j 2
```

#### Installing `python` wrapper

```
cd python
python ../../setup.py build --build-dir=.. --skip-build install
```

#### Testing it

From the root directory (`dynet-base/dynet`) of the `dynet` install, run this command:
```
python examples/python/xor.py
```

### `Windows` installation

I have not tried installing for `Windows`, but instructions are [here](http://dynet.readthedocs.io/en/latest/python.html#windows-support)
