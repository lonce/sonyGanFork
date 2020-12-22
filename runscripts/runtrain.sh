#!/bin/bash
# You should run this script in a docker container, such as:
#docker run --ipc=host --gpus '"device=0"' -it -v $(pwd):/sonyGan -v /home/lonce/3tb/DATA_OTHERS/nsynth-train.jsonwav:/mydata lonce/sonygan:2020.12.16

configfile=$1

#########################################################################
D=$(date +%Y.%m.%d_%H.%M.%S) #'%(%Y-%m-%d)T'
WD=${PWD##*/}

logdir=logs/logs_$D
mkdir -p $logdir
echo "copying your config file, $1, to your log dir: $logdir"
echo "listing of mydata: "
ls /mydata
cp $1 $logdir  # copy the config file into the log directory 
#printf 'logdir is %s\n' "$logdir"
python train.py --restart -c $configfile -s 5000 -l 5000 > $logdir/stdout.txt 2>$logdir/stderr.txt &
echo "tail -f $logdir/stderr.txt  (to watch the river flow)"

