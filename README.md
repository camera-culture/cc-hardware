# Camera Culture Hardware Repo

This is a monorepo for all hardware scripts used in Camera Culture.

## Getting Started

To install the `cc_hardware` package, you will need the `poetry` package.
To install, run the following command:

```bash
pip install poetry
```

You can then install the `cc_hardware` package by running the following commands:

```bash
# Clone the repository
git clone git@github.com:camera-culture/cc_hardware.git
cd cc_hardware

# Install the package
poetry install
```

`poetry install` will install the package and it's dependencies in develop mode.
This allows you to make changes to the code and have them reflected in the package
without having to reinstall the package (i.e. a symlink is created to the package).
You can also install `cc_hardware` with `pip install .` if you don't need this
behavior.

## Repo Structure

`cc_hardware` is a monorepo that contains multiple packages. Each package is a
subdirectory within the `pkgs` directory and should be installed as a separate package
(done automatically with `poetry install`/`pip install .` as described in
[Getting Started](#getting-started)).

The current supported packages are as follows:

- [`algos`](https://camera-culture.github.io/cc-hardware/usage/api/algos):
    Contains algorithms for processing data.
- [`drivers`](https://camera-culture.github.io/cc-hardware/usage/api/drivers):
    Contains drivers for interfacing with hardware.
- [`utils`](https://camera-culture.github.io/cc-hardware/usage/api/utils):
    Contains utility functions and classes.
- [`tools`](https://camera-culture.github.io/cc-hardware/usage/cli):
    Contains tools for working with hardware, such as calibration or visualization
    scripts.

### Package Structure

Each package is structured as follows:

```
pkgs/
    <package_name>/
        README.md
        poetry.toml
        pyproject.toml
        cc_hardware/
            <package_name>/
                __init__.py
                ...
```

In this way, when installing each package, the import path will be
`cc_hardware.<package_name>.<module_name>`.
