from setuptools import setup, find_packages

setup(
    name="geo-edit",
    version="0.1.0",
    package_dir={"": ".."},
    packages=[p for p in find_packages("..") if p == "geo_edit" or p.startswith("geo_edit.")],
)
