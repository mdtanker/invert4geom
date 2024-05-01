from __future__ import annotations

import invert4geom

project = "invert4geom"
copyright = "2023, Matt Tankersley"
author = "Matt Tankersley"
version = release = invert4geom.__version__
extensions = [
    "sphinx.ext.autodoc",  # needed for typehints
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx_design",
    "nbsphinx",
    "sphinxcontrib.bibtex",
]
source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

bibtex_bibfiles = ["_invert4geom_refs.bib"]

nbsphinx_execute = "never"

myst_enable_extensions = [
    "colon_fence",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    #
    # Runtime deps
    #
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "verde": ("https://www.fatiando.org/verde/latest/", None),
    "rioxarray": ("https://corteva.github.io/rioxarray/stable/", None),
    # pykdtree
    "xrft": ("https://xrft.readthedocs.io/en/stable/", None),
    "harmonica": ("https://www.fatiando.org/harmonica/latest/", None),
    "polartoolkit": ("https://polartoolkit.readthedocs.io/en/latest/", None),
    "numba": ("https://numba.pydata.org/numba-doc/latest/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    # nptyping
    # numba_progress
    # "tqdm": ("https://tqdm.github.io/", None),
    "pygmt": ("https://www.pygmt.org/latest/", None),
    #
    # Viz deps
    #
    "pyvista": ("https://docs.pyvista.org/", None),
    # trame
    # ipywidgets
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    #
    # Opti deps
    #
    "optuna": ("https://optuna.readthedocs.io/en/stable/", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

always_document_param_types = True
add_module_names = False
add_function_parentheses = False

# API doc configuration
# -----------------------------------------------------------------------------
# autosummary_generate = True
# autodoc_default_options = {
#     "members": True,
#     "show-inheritance": True,
# }
# apidoc_module_dir = '../src/invert4geom'
# apidoc_excluded_paths = ['tests']
# apidoc_separate_modules = False
autoapi_dirs = ["../src/invert4geom"]
autoapi_type = "python"
autoapi_add_toctree_entry = False
autodoc_typehints = "description"

# HTML output configuration
# -----------------------------------------------------------------------------
html_title = f'{project} <span class="project-version">{version}</span>'
# html_logo = "_static/harmonica-logo.png"
# html_favicon = "_static/favicon.png"
html_last_updated_fmt = "%b %d, %Y"
html_copy_source = True
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = False
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/mdtanker/invert4geom",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "home_page_in_toc": False,
}
