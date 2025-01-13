# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import sys
from datetime import date
from functools import partial

# The full version, including alpha/beta/rc tags
from importlib.metadata import version as get_version
from pathlib import Path

root_path = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, root_path)
sys.setrecursionlimit(1500)

# -- Project information -----------------------------------------------------

project = "cc_hardware"
copyright = f"{date.today().year}, Camera Culture, MIT Media Lab"
author = "Camera Culture, MIT Media Lab"
release = get_version(project)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    # "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "autoapi.extension",
    "myst_parser",
    "sphinxarg.ext",
    "sphinxcontrib.googleanalytics",
    "sphinx.ext.inheritance_diagram",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


def _prep_jinja_env(jinja_env):
    jinja_env.tests["loc_file"] = lambda name: Path(name.strip("/")).exists()


autoapi_prepare_jinja_env = _prep_jinja_env


def copy_readmes_to_autoapi(app, exception):
    """
    Copy README.md files from each package in `../pkgs` to the corresponding
    directory in the autoapi output.
    """
    from shutil import copyfile

    if exception is not None:
        return

    pkgs_dir = Path(root_path) / "pkgs"
    destination_root = Path(root_path) / "docs" / autoapi_root

    # Ensure the destination root exists
    destination_root.mkdir(parents=True, exist_ok=True)

    for package_dir in pkgs_dir.iterdir():
        if package_dir.is_dir():  # Only process directories (packages)
            source_readme = package_dir / "README.md"
            destination_dir = destination_root / project / package_dir.name
            destination_readme = destination_dir / "README.md"

            if source_readme.exists():
                destination_dir.mkdir(parents=True, exist_ok=True)
                copyfile(source_readme, destination_readme)
                print(f"Copied {source_readme} to {destination_readme}")
            else:
                print(f"WARNING: README.md not found in {package_dir}")


def update_inits(*args, create=False, remove=False):
    # This method will recursively find package dirs and create an init file under
    # package_dir / project. If create is False, it will remove the init files.
    assert create != remove, "Cannot create and remove inits at the same time"

    pkgs_dir = Path(root_path) / "pkgs"
    for package_dir in pkgs_dir.iterdir():
        if package_dir.is_dir():
            init_file = package_dir / project / "__init__.py"
            if create:
                init_file.touch()
                print(f"Created {init_file}")
            if remove:
                assert init_file.exists(), f"{init_file} does not exist"
                init_file.unlink()
                print(f"Removed {init_file}")


def setup(app):
    update_inits(create=True)
    app.connect("build-finished", copy_readmes_to_autoapi)
    app.connect("build-finished", partial(update_inits, remove=True))


# autoapi config
autoapi_type = "python"
autoapi_dirs = ["../pkgs/"]
autoapi_options = [
    "members",
    "show-inheritance-diagram",
    "show-module-summary",
    "special-members",
    "imported-members",
    # "inherited-members"  # Doesn't work with viewcode extension
]
autoapi_ignore = ["*/_templates/*", "*/usage/api/*"]
autoapi_keep_files = True
# autoapi_keep_files = False
autoapi_generate_api_docs = True
autoapi_add_toctree_entry = False
autoapi_root = "usage/api/"
autoapi_template_dir = "_templates"
# autoapi_member_order = "groupwise"

# suppress_warnings = ["autoapi"]

autodoc_typehints = "none"

add_module_names = False

# Napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# MyST
myst_heading_anchors = 7

viewcode_enable_epub = True

# Display todos by setting to True
todo_include_todos = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["env"]

# Google Analytics
googleanalytics_id = "G-WK44DH2ZCM"

inheritance_edge_attrs = dict(color="gray")

# -- Options for HTML output -------------------------------------------------

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_favicon = "_static/cc-transparent.png"
html_theme_options = {
    "announcement": """
        <a style=\"text-decoration: none; color: white;\"
           href=\"https://www.media.mit.edu/groups/camera-culture/overview/\">
           <img src=\"/cc-hardware/_static/cc-transparent.png\"
                style=\"
                    vertical-align: middle;
                    display: inline;
                    padding-right: 7.5px;
                    height: 20px;
                \"/>
           Checkout Camera Culture!
        </a>
    """,
    "sidebar_hide_name": True,
    "light_logo": "cc-white-bg.png",
    "dark_logo": "cc-transparent.png",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/custom.css",
]
