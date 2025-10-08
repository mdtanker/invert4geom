import importlib.metadata

project = "invert4geom"
copyright = "2023, Matt Tankersley"
author = "Matt Tankersley"
version = release = importlib.metadata.version("invert4geom")
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "sphinxcontrib.bibtex",
    "sphinx_design",
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
    "numba": ("https://numba.readthedocs.io/en/stable/index.html", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    # numba_progress
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
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
    ("py:class", "optional"),
    ("py:class", "optuna.trial"),
    ("py:class", "optuna.study"),
    ("py:class", "optuna.storages.BaseStorage"),
    ("py:class", "plotly.graph_objects.Figure"),
    ("py:class", "Ellipsis"),
]

always_document_param_types = True
add_module_names = False
add_function_parentheses = False

nbsphinx_execute = "never"

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'png2x'}",
    "--InlineBackend.rc=figure.dpi=96",
]

nbsphinx_kernel_name = "python3"

# HTML output configuration
# -----------------------------------------------------------------------------
html_title = f'{project} <span class="project-version">{version}</span>'
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
