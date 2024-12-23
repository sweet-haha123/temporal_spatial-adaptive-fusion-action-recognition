#!/usr/bin/env bash

cd ../
python build_rawframes.py /media/gaocq/02599ee7-b788-4c6d-a01a-dc083eb48162/qyh/ntu_dataset/ir_data/videos/ /media/gaocq/02599ee7-b788-4c6d-a01a-dc083eb48162/qyh/ntu_dataset/ir_data/rawframes/ --task rgb --level 2 --ext avi --use-opencv
echo "Genearte raw frames (RGB only)"

#cd ucf101/
