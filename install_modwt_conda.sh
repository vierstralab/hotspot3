#!/bin/bash
set -e
# Check if g++ is installed
if ! which g++ > /dev/null; then
    echo "Error: g++ is not installed. Please install g++ and try again."
    exit 1
fi

# Check if make is installed
if ! which make > /dev/null; then
    echo "Error: make is not installed. Please install make and try again."
    exit 1
fi

github_link="https://github.com/wishabc/modwt"
echo "Installing modwt from $github_link into conda env - $CONDA_PREFIX"

git clone $github_link
cd modwt

make -C .

mkdir -p $CONDA_PREFIX/bin
cp bin/modwt $CONDA_PREFIX/bin/
cp bin/modwt.so $CONDA_PREFIX/bin/

echo "modwt installed successfully to $CONDA_PREFIX/bin/"

echo "Cleaning up..."
cd ..
rm -rf modwt

