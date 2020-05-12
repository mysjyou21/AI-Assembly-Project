#!/bin/sh

inputdir=./assemblies_300dpi
outputdir=./output
resultdir=./results

for input in $inputdir/*
do
    filename=${input:20}
    argument="--assembly_name=${filename} --cutpath=${inputdir} --resultpath=./intermediate_results --ocr_modelpath=./function/OCR/weight --csv_dir=./output --eval_print=False"
    python main.py $argument
done
