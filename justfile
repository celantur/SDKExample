sdk_install_dir := "/usr/local/lib/cmake"

default:
    just --list

build:
    #!/bin/bash
    set -e
    mkdir -p build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release \
    	  -GNinja \
    	  -DCppProcessing_DIR={{sdk_install_dir}} \
    	  ..
    ninja

    echo {{GREEN}}DONE!{{NORMAL}}


