# Contributing

Contributing to the repository is fairly easy, so please see below for instructions on how to actually make contributions to this project.

## Setup

There are two forms of contributions: source code or documentation. Editing the documentation is as simple as cloning the repo and adding/editing content within the `docs` folder. All documentation is written in `markdown` and converted to `html` through `myst_parser` and `sphinx`. To edit the source code, as well as the documentation, you will want to install the package through a symlink.

```{note}
A `conda` or `virtualenv` will add isolation to your python environments and reduce conflicts amongst packages. It is _highly_ recommended to use one!!
```

### Cloning the Repo

Clone the repo as normal:

```bash
git clone https://github.com/camera-culture/cc-hardware.git && cd cc-hardware
```

### Installing

Since the package is [organized as a monorepo](#file-structure), you will need to install the package using `poetry`. The code will be installed in develop mode (i.e. as a symbolic link) so that changes to the source code are immediately reflected in the package.

```bash
pip install poetry
poetry install
```

### Deploy your Changes

[GitHub actions](https://github.com/features/actions) are used to automatically build the site and [GitHub pages](https://pages.github.com/) are used to host the static site. To update deployed content, you have to push to the `main` branch. Once the changes are pushed, the site will rebuild. Please ensure there are no errors in your code/documentation before doing so, as you may get an email from github if something bad happens.

<!-- Further, to update the package available on [PyPI](https://pypi.org/project/EyesOfCambrian/), you must create a [git tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging). When a tag is created and pushed to GitHub, it will start an Action which will automatically push the new release to PyPI. See [Versioning](#versioning) for information on how versioning works with `EyesOfCambrian`. The Github Action only runs when the tag is pushed to main through a merge request. To create a tag, you may do the following: -->
<!--  -->
<!-- ```bash -->
<!-- git tag <tag> # An example tag is v3.0.1 -->
<!-- git push origin main --tags -->
<!-- ``` -->

<!-- #### Versioning

Versioning is done automatically through `tags` by [setuptools_scm](https://github.com/pypa/setuptools_scm). When a tag is pushed to the `main` branch, a new package is pushed to PyPI with the attached tag. Therefore, you must ensure the tag you push is _after_ the previous tags seen on GitHub (otherwise nothing will push to PyPI). -->

## Guidelines

### File Structure

The repository is structured as follows:

```
cc-hardware/
├── pkgs/
│   ├── algos/          # Contains algorithms for processing data
│   ├── drivers/        # Contains drivers for interfacing with hardware
│   ├── utils/          # Contains utility functions and classes
│   ├── tools/          # Contains tools for working with hardware, such as calibration or visualization scripts
│   └── .../            # Additional packages
├── docs/               # Contains documentation
├── LICENSE
└── pyproject.toml      # Package description and installation instructions for poetry
```

### Editing the Source Code

If you plan on editing the source code, please visit the corresponding package under the `pkgs/` folder. Each package should be structured as follows:

```
<pkg>/
├── README.md
├── poetry.toml
├── pyproject.toml
└── cc_hardware/
    └── <pkg>/
        ├── __init__.py
        └── ...
```

#### Commenting

Please follow [Google's guidelines for Python Styling](https://google.github.io/styleguide/pyguide.html). These comments are also used to automatically generate the documentation.

```{note}
Any docstrings parsed by `autosimple`, such as the functions in [usage.md](./usage/index.md), are parsed as markdown. Docstrings parsed by autoapi, such as in [cambrian.envs.MjCambrianEnv](./usage/api/cambrian/envs/env/index), are parsed as reStructuredText.
```

### Editing the Documentation

If you plan on editing the documentation pages (i.e. adding a tutorial or fixing an existing page), please see the `docs/` folder. The `docs/` folder is structured as follows:

```
docs/
├── _static/            # Static files that persist through the build process
│   ├── css/custom.css  # Custom css changes that are different from the default furo theme
│   └── ...             # Images, favicons, etc.
├── usage/              # Usage reference guide for the EyesOfCambrian package
│   └── ...
├── conf.py             # Settings related to extensions, themes, etc.
└── index.md            # The "home" page
```

Please try to maintain the file structure as described above. All tabs with only a single page (i.e. background or contributing), should have their `markdown` file with in the `docs/` folder. If the tab has or will have multiple pages (i.e. a tutorials tab), create a folder titled the same as that tab. To add pages, insert the name of the file without the `.md` extension within the table of contents inside `index.md`. Each folder should also contain an `index.md` used as the home page of that tab.

Markdown files are converted to reStructuredText by `myst_parser` which is used by the documentation package [Sphinx](https://www.sphinx-doc.org/en/master/). Both Markdown and reStructuredText have their advantages and disadvantages, `myst_parser` allows us to use the easy to understand `markdown` language but also compile the content down to something Sphinx understands. To see additional features of the `myst_parser` library, please visit their [website](https://myst-parser.readthedocs.io/en/latest/).

## Building the Documentation

There are multiple ways to build sphinx documentation. The easiest is using the `Makefile` or `make.bat` file provided directly in this repository. You will need to install all the necessary dependencies and build the html pages. To do that, run the following commands:

```bash
cd cc-hardware/
poetry install --with dev --with docs
cd docs/
make clean html
```

To view the build, go to your browser, and open the `index.html` file located inside `docs/build/html/`.

`sphinx-autobuild` is also extremely easy to use and will automatically build the html pages when a change is made. See their [PyPI page](https://pypi.org/project/sphinx-autobuild/). TL;DR: `pip install sphinx-autobuild` and run `sphinx-autobuild . build`.
