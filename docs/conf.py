# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Beam-DS'
copyright = '2024, Elad Sarafian'
author = 'Elad Sarafian'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings.
extensions = [
            'sphinx.ext.autodoc',  # Include documentation from docstrings
            'sphinx.ext.napoleon',  # Support for Google-style docstrings
            'sphinx.ext.viewcode',  # Add links to source code
            'sphinx.ext.githubpages',  # Publish HTML files to GitHub Pages
             ]



templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


# The theme to use for HTML and HTML Help pages.
html_theme = 'pydata_sphinx_theme'

# Configuration for the theme
html_theme_options = {
            "github_url": "https://github.com/mlutils/beamds",
                "use_edit_page_button": True,
                    "show_prev_next": False
                    }

# Optionally set the base URL for the documentation, which will require this to be set if you use GitHub Pages, for example.
# html_baseurl = 'https://your-username.github.io/your-repo/'


html_theme_options = {
            "use_edit_page_button": True,
                "show_prev_next": False
                }

html_context = {
            "github_user": "mlutils",  # Your GitHub username or organization
                "github_repo": "beamds",  # Your repository
                    "github_version": "main",  # The branch containing your docs
                        "doc_path": "docs",  # Path within the repository to your documentation source
                        }

