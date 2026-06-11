# ── Sphinx Configuration for HAP Framework ─────────────────────
# Run: sphinx-build -b html docs/ docs/_build/
# Or:  make -C docs/ html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Hyperdimensional Active Perception (HAP)"
copyright = "2024, Arthedain AI"
author = "Arthedain AI"
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "myst_parser",
]

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_ivar = True

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# MyST parser for Markdown
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# Theme
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}
html_static_path = []

# Don't show module paths in autosummary
add_module_names = False