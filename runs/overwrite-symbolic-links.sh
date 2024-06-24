
cd "$(dirname "${BASH_SOURCE[0]}")"
echo "Current directory: $(pwd)"

ln -s -f ../src/beam beam

cd ../tests
echo "Current directory: $(pwd)"
ln -s -f ../src/beam beam

cd ../examples
echo "Current directory: $(pwd)"
ln -s -f ../src/beam beam

cd ../
echo "Current directory: $(pwd)"
ln -s -f src/beam beam

