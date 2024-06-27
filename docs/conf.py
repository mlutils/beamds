# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = 'Beam-DS'
copyright = '2024, Elad Sarafian'
author = 'Elad Sarafian'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',  # Include documentation from docstrings
    'sphinx.ext.napoleon',  # Support for Google-style docstrings
    'sphinx.ext.viewcode',  # Add links to source code
    'sphinx.ext.githubpages',  # Publish HTML files to GitHub Pages
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']  # Ensure this directory exists if you use it

html_theme_options = {
    "github_url": "https://github.com/mlutils/beamds",
    "use_edit_page_button": True,
    "show_prev_next": False,
    "logo": {
            "image_light": "_static/logo.png",  # Path to the light mode logo
            "image_dark": "_static/logo.png"    # Path to the dark mode logo, if different
        }
}

html_context = {
    "github_user": "mlutils",
    "github_repo": "beamds",
    "github_version": "main",
    "doc_path": "docs",
}

