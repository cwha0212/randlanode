cd src/utils/nearest_neighbors
python setup.py install --home="."
cd ../../../

cd src/utils/cpp_wrappers
sh compile_wrappers.sh
cd ../../../