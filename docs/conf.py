# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'fast_ensemble'
copyright = '2024, Flavia Maria Galeazzi, Pablo Arantes, Gabriel Monteiro Da Silva, Isabel Varghese, Ananya Shukla, Brenda M. Rubenstein'
author = 'Flavia Maria Galeazzi, Pablo Arantes, Gabriel Monteiro Da Silva, Isabel Varghese, Ananya Shukla, Brenda M. Rubenstein'
release = '2024'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'm2r',
]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
napoleon_google_docstring = True
source_suffix = ['.rst', '.md']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
