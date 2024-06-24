cd "$(dirname "${BASH_SOURCE[0]}")"
cd ../
rm -f beam

cd runs
echo "Current directory: $(pwd)"
ln -s -f ../src/beam beam

cd ../tests
echo "Current directory: $(pwd)"
ln -s -f ../src/beam beam

cd ../examples
echo "Current directory: $(pwd)"
ln -s -f ../src/beam beam

cd ../notebooks
echo "Current directory: $(pwd)"
ln -s -f ../src/beam/setup beam_setup

cd ../
echo "Current directory: $(pwd)"
ln -s -f src/beam beam

