
cd "$(dirname "${BASH_SOURCE[0]}")"

ln -s -f ../src/beam beam

cd ../tests
ln -s -f ../src/beam beam

cd ../examples
ln -s -f ../src/beam beam

cd ../
ln -s -f src/beam beam

