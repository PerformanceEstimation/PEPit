import os
import sys
import shutil

# The module you're documenting (assumes you've added the project root dir to sys.path)
sys.path.insert(0, os.path.abspath('../..'))
import PEPit

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
    'myst_nb',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
    '.ipynb': 'myst-nb',
}

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']
templates_path = ['_templates']

# -- MyST & Notebook Settings ------------------------------------------------
nb_execution_mode = "off"
myst_title_to_header = True
myst_dmath_allow_labels = False    # Disable MyST labels to stop double-numbering
myst_update_mathjax = False        # Let MathJax 3 handle its own configuration
myst_dmath_double_inline = True

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
    "html_image",
    "html_admonition",
]

# -- MathJax 3 Settings ------------------------------------------------------
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
        'processEscapes': True,
        'tags': 'ams',             # AMS numbering (1), (2) on the right
        'useLabelIds': True
    },
    'options': {
        'processHtmlClass': 'tex2jax_process|mathjax_process|math|output_area',
    }
}

# -- HTML Output Settings ---------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_secnumber_suffix = " "

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False

# -- Napoleon & Autodoc -----------------------------------------------------
napoleon_custom_sections = [('Returns', 'params_style'), ('Attributes', 'params_style')]
autoclass_content = 'both'
autodocgen_config = [{
    'modules': [PEPit],
    'generated_source_dir': './autodocgen-output/',

    # Skips any module with "__" in the name or specific internal modules
    'skip_module_regex': '(.*[.]__|myskippedmodule)',

    # Log file to track exactly what is being documented
    'write_documented_items_output_file': 'autodocgen_documented_items.txt',

    # Custom options for specific classes if needed
    'autodoc_options_decider': {
        'PEPit.FooBar': {'inherited-members': True},
    },

    # Renames the top-level module to "API Reference" in the TOC
    'module_title_decider': lambda modulename: 'API Reference' if modulename == 'PEPit' else modulename,
}]

# -- Automated Notebook Copying ---------------------------------------------
current_dir = os.path.dirname(__file__)
nb_source = os.path.abspath(os.path.join(current_dir, '../../ressources/demo/'))
nb_dest = os.path.join(current_dir, 'notebooks_folder')

if os.path.exists(nb_dest):
    shutil.rmtree(nb_dest)
os.makedirs(nb_dest)

if os.path.exists(nb_source):
    for file in os.listdir(nb_source):
        if file.endswith('.ipynb'):
            shutil.copy2(os.path.join(nb_source, file), os.path.join(nb_dest, file))


# -- Setup Function ---------------------------------------------------------
def setup(app):
    app.add_css_file('custom.css')
