# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "WEFE"
copyright = "2025, The WEFE Team"
author = "The WEFE Team"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
from wefe import __version__  # noqa: E402

version = __version__
release = __version__
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Minimum Sphinx version required
needs_sphinx = "4.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
]

# Configure autodoc
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Generate autosummary even if no references
autosummary_generate = True

# NumPy-style docstring configuration
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Math rendering configuration
if os.environ.get("NO_MATHJAX"):
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
    mathjax_path = ""
else:
    mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"
    mathjax_config = {
        "tex": {
            "inlineMath": [["$", "$"], ["\\(", "\\)"]],
            "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        }
    }

# Source file configuration
templates_path = ["_templates"]
source_suffix = {
    ".rst": "restructuredtext",
}
master_doc = "index"
exclude_patterns = ["_build", "_templates", "Thumbs.db", ".DS_Store"]

# Internationalization
language = "en"

# Syntax highlighting
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# HTML theme configuration
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": True,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# HTML output configuration
html_title = f"WEFE v{version} Documentation"
html_short_title = "WEFE Documentation"
html_logo = "logos/WEFE_2_BLANCO.svg"
html_favicon = "logos/WEFE.ico"

# Static files and styling
html_static_path = ["_static"]
html_css_files = [
    "css/theme_overrides.css",
]

# JavaScript files
html_js_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"
]

# Additional HTML options
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True
html_copy_source = True
html_use_opensearch = ""

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": r"""
        \usepackage{amsmath}
        \usepackage{amsfonts}
        \usepackage{bm}
        \usepackage{morefloats}
        \usepackage{enumitem}
        \setlistdepth{10}
        \let\oldhref\href
        \renewcommand{\href}[2]{\oldhref{#1}{\hbox{#2}}}
    """,
}

latex_documents = [
    (master_doc, "WEFE.tex", "WEFE Documentation", "WEFE Team", "manual"),
]

# -- Options for manual page output ------------------------------------------

man_pages = [(master_doc, "wefe", "WEFE Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "WEFE",
        "WEFE Documentation",
        author,
        "WEFE",
        "Word Embedding Fairness Evaluation (WEFE) is an open source library for "
        "measuring and mitigating bias in word embedding models.",
        "Miscellaneous",
    ),
]

# -- Options for EPUB output -------------------------------------------------

epub_show_urls = "footnote"

# -- Extension-specific configuration ----------------------------------------

# Intersphinx configuration for cross-references to other projects
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "gensim": ("https://radimrehurek.com/gensim/", None),
}

# Sphinx Gallery configuration
sphinx_gallery_conf = {
    "doc_module": "wefe",
    "backreferences_dir": "generated",
    "reference_url": {"wefe": None},
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"\.py$",
    "ignore_pattern": r"__init__\.py",
    "download_all_examples": False,
    "plot_gallery": "True",
}

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True

# TODO extension
todo_include_todos = False

# Output file base name for HTML help builder
htmlhelp_basename = "wefe"


def setup(app) -> None:
    """Sphinx setup hook."""
    # Add custom CSS for theme overrides
    app.add_css_file("css/theme_overrides.css")
