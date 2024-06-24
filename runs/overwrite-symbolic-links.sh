cd "$(dirname "${BASH_SOURCE[0]}")"
cd ../
rm -f beam

cd runs
echo "Current directory: $(pwd)"
rm -f beam
ln -s ../beam beam

cd ../tests
echo "Current directory: $(pwd)"
rm -f beam
ln -s ../beam beam

cd ../examples
echo "Current directory: $(pwd)"
rm -f beam
ln -s ../beam beam

cd ../notebooks
echo "Current directory: $(pwd)"
rm -f beam_setup
ln -s -f ../beam/setup beam_setup

