# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import importlib
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------

project = "grid"
copyright = "2024, QC-Devs"
author = "QC-Devs"
html_logo = "./images/grid_logo_website.svg"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "nbsphinx_link",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.linkcode",
]

# List of arguments to be passed to the kernel that executes the notebooks:
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc=figure.dpi=200",
]
# nbsphinx_input_prompt = 'In [%s]:'
# nbsphinx_output_prompt = 'Out[%s]:'
# explicitly dis-/enabling notebook execution
nbsphinx_execute = "never"

# Don't add .txt suffix to source files:
html_sourcelink_suffix = ""

# This is processed by Jinja2 and inserted before each notebook

# nbsphinx_prolog = r"""
# nbsphinx_epilog = r"""
# {% set docname = env.docname.split("/")[-1] %}
# .. raw:: html
# .. role:: raw-html(raw)
#  :format: html
# .. nbinfo::
#  The corresponding file can be obtained from:
#  - Jupyter Notebook: :download:`{{docname+".ipynb"}}`
#  - Interactive Jupyter Notebook: :raw-html:`<a href="https://mybinder.org/v2/gh/theochem/procrustes/master?filepath=doc%2Fnotebooks%2F/{{ docname }}.ipynb"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>`    """


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_style = "css/override.css"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "collapse_navigation": False,
    "logo_only": True
}


# -- Configuration for autodoc extensions ---------------------------------

autodoc_default_options = {
    "undoc-members": True,
    "show-inheritance": True,
    "members": None,
    "inherited-members": True,
    "ignore-module-all": True,
}


add_module_names = False


def autodoc_skip_member(app, what, name, obj, skip, options):
    """Decide which parts to skip when building the API doc."""
    if name == "__init__":
        return False
    if name.startswith("test_"):
        return False
    return skip


def setup(app):
    """Set up sphinx."""
    app.connect("autodoc-skip-member", autodoc_skip_member)


# -- Link to Github ------------------------------
# Configure viewcode extension.
code_url = f"https://github.com/theochem/grid/blob/master/src/grid"


def linkcode_resolve(domain, info):
    # Non-linkable objects from the starter kit in the tutorial.
    if domain != 'py':
        return None
    if not info['module']:
        return None

    mod = importlib.import_module(info["module"])
    if "." in info["fullname"]:
        objname, attrname = info["fullname"].split(".")
        obj = getattr(mod, objname)
        try:
            # object is a method of a class
            obj = getattr(obj, attrname)
        except AttributeError:
            # object is an attribute of a class
            return None
    else:
        obj = getattr(mod, info["fullname"])
    try:
        file = inspect.getsourcefile(obj)
        lines = inspect.getsourcelines(obj)
    except TypeError:
        # e.g. object is a typing.Union
        return None
    file = os.path.relpath(file, os.path.abspath(".."))

    # Get the file name e.g. /usr/conda/py37/grid/angular.py -> angular.py
    split = file.split("grid/")
    file = split[-1]

    start, end = lines[1], lines[1] + len(lines[0]) - 1

    return f"{code_url}/{file}#L{start}-L{end}"
