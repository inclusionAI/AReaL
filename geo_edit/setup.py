from setuptools import setup, find_packages

setup(
    name="geo-edit",
    version="0.1.0",
    package_dir={"geo_edit": "."},
    packages=["geo_edit"] + ["geo_edit." + p for p in find_packages(".")],
)
