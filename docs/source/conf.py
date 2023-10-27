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
# import os
import os
import sys

# The module you're documenting (assumes you've added the project root dir to sys.path)
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'PEPit'
copyright = '2021, PEPit Contributors'
author = 'PEPit Contributors'

# The full version, including alpha/beta/rc tags
release = '0.0.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # 'easydev.copybutton',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    # 'sphinxcontrib_autodocgen',
    'myst_parser',
]

napoleon_custom_sections = [('Returns', 'params_style'),
                            ('Attributes', 'params_style')]

import PEPit

autodocgen_config = [{
    'modules': [PEPit],
    'generated_source_dir': './autodocgen-output/',

    # if module matches this then it and any of its submodules will be skipped
    'skip_module_regex': '(.*[.]__|myskippedmodule)',

    # produce a text file containing a list of everything documented. you can use this in a test to notice
    # when you've intentionally added/removed/changed a documented API
    'write_documented_items_output_file': 'autodocgen_documented_items.txt',

    # customize autodoc on a per-module basis
    'autodoc_options_decider': {
        'mymodule.FooBar': {'inherited-members': True},
    },

    # choose a different title for specific modules, e.g. the toplevel one
    'module_title_decider': lambda modulename: 'API Reference' if modulename == 'mymodule' else modulename,
}]

autoclass_content = 'both'

# Include or not the special methods
napoleon_include_special_with_doc = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# # Make the copy paste possible for any example code in documentation
# import easydev
#
# jscopybutton_path = easydev.copybutton.get_copybutton_path()
#
# # if not os.path.isdir('_static'):
# #     os.mkdir('_static')
#
# import shutil
#
# shutil.copy(jscopybutton_path, '_static')

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
