#!/usr/bin/env bash

cd ../../../

PYTHONPATH=. python tools/data/build_file_list.py ir_data /media/gaocq/02599ee7-b788-4c6d-a01a-dc083eb48162/qyh/ntu_dataset/ir_data/rawframes/ --level 2 --format rawframes --shuffle
echo "Filelist for rawframes generated."

cd tools/data/ucf101/
