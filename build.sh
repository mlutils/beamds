#!/bin/bash

# Get version from _version.py
# the version is defined in beam./_version.py
# the line to invoke is:
# __version__ = '2.8.0b'
# use sed to extract the version number
VERSION=$(python -c "from beam._version import __version__; print(__version__)")

# Update version in pyproject.toml
sed -i '' 's/^version = ".*"/version = "'"$VERSION"'"/' pyproject.toml

echo "Version updated to $VERSION"

# update the version for poetry
poetry version "$VERSION"

# Build the package
poetry build 