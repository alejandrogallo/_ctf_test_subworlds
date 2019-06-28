set -eu

folder=ctf

git clone https://github.com/cyclops-community/ctf $folder &&
cd $folder &&
./configure 'CXX=mpiicc' &&
git checkout v1.5.4 &&
make -j 4 ctflib
