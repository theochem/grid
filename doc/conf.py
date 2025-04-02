# Configuration file for the Sphinx documentation builder.
#
# Official documentation: https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib
import inspect
import os
import sys

# -- Path setup --------------------------------------------------------------
# Ensure project modules can be found by Sphinx
BASE_DIR = os.path.abspath("..")
sys.path.insert(0, BASE_DIR)

# -- Project Information -----------------------------------------------------
project = "grid"
copyright = "2024, QC-Devs"
author = "QC-Devs"
html_logo = "./images/grid_logo_website.svg"

# -- Sphinx Extensions -------------------------------------------------------
# Add Sphinx extensions to enable features like autodoc, math rendering, and Jupyter notebook integration.
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

# -- Jupyter Notebook Settings for Documentation -----------------------------
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc=figure.dpi=200",
]
nbsphinx_execute = "never"  # Prevent execution of notebooks during build

# -- Source Files & Template Settings ----------------------------------------
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- HTML Output Configuration -----------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_style = "css/override.css"

html_theme_options = {
    "collapse_navigation": False,
    "logo_only": True,
}

# -- Autodoc Configuration ---------------------------------------------------
autodoc_default_options = {
    "undoc-members": True,
    "show-inheritance": True,
    "members": None,
    "inherited-members": True,
    "ignore-module-all": True,
}

add_module_names = False  # Prevents redundant module names in generated docs

def autodoc_skip_member(app, what, name, obj, skip, options):
    """
    Control which members are included in the generated documentation.

    - Always document `__init__` methods.
    - Include test functions (e.g., `test_*`).
    """
    if name == "__init__":
        return False
    if name.startswith("test_"):
        return False
    return skip

def setup(app):
    """Configure Sphinx extensions and event handlers."""
    app.connect("autodoc-skip-member", autodoc_skip_member)

# -- GitHub Link for Source Code ---------------------------------------------
# Configure the 'viewcode' extension for linking to the GitHub repository.
GITHUB_REPO_URL = "https://github.com/theochem/grid/blob/master/src/grid"

def linkcode_resolve(domain, info):
    """
    Generate GitHub links for source code.

    - Handles both module-level and class-level objects.
    - Ensures proper error handling for missing or unrecognized objects.
    """
    if domain != "py" or not info["module"]:
        return None

    try:
        mod = importlib.import_module(info["module"])
        obj = mod
        for part in info["fullname"].split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        
        file_path = inspect.getsourcefile(obj)
        source_lines, start_line = inspect.getsourcelines(obj)
        end_line = start_line + len(source_lines) - 1

        if not file_path:
            return None

        # Normalize the path to match the repository structure
        file_relpath = os.path.relpath(file_path, BASE_DIR).replace("\\", "/")

        return f"{GITHUB_REPO_URL}/{file_relpath}#L{start_line}-L{end_line}"
    
    except (ImportError, AttributeError, TypeError):
        return None
