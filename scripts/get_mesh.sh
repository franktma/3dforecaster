#!/bin/bash
if [ $# -lt 1 ]; then
   echo Usage: $0 [input list] [output dir]
   exit -1
fi

filelist=$1  #validation.list
outdir=$2

for i in `cat $filelist `; do cp -i /Volumes/work/data/paperlessparts/Thingi10K/raw_meshes/$i $2 ; done
