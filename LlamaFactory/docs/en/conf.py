import os
import sys


# Add parent dir to path to allow importing conf.py
sys.path.insert(0, os.path.abspath(".."))

from conf import *  # noqa: F403


# Language settings
language = "en"
html_search_language = "en"

# Static files
# Point to the root _static directory
html_static_path = ["../_static"]

# Add custom JS for language switcher
html_js_files = [
    "js/switcher.js",
]
