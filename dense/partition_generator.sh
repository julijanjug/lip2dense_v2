#!/bin/bash

dir_size=3000
dir_name="../images_"
source_dir="./LIP/LIP_dataset/valid_set/images_copy/"

cd $source_dir
n=$((`find . -maxdepth 1 -type f | wc -l`/$dir_size+1))
echo $PWD
for i in `seq 1 $n`;
do
    mkdir -p "$dir_name$i";
    find . -maxdepth 1 -type f | head -n $dir_size | xargs -i mv "{}" "$dir_name$i"
done