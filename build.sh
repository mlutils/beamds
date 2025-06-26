#!/bin/bash

# Get version from _version.py
# the version is defined in beam./_version.py
# the line to invoke is:
# __version__ = '2.8.0b'
# use sed to extract the version number

#VERSION=$(python -c "from beam._version import __version__; print(__version__)")
# get the version from the _version.py file without importing it
VERSION=$(sed -n "s/__version__ = ['\"]\([^'\"]*\)['\"]/\\1/p" beam/_version.py)

# Update version in pyproject.toml
sed -i '' 's/^version = ".*"/version = "'"$VERSION"'"/' pyproject.toml

echo "Version updated to $VERSION"

# update the version for poetry
poetry version "$VERSION"

# Build the package
poetry build

# echo instructions how to deploy the package with poetry
echo "Package built successfully. To deploy the package, run poetry publish"