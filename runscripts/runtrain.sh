#!/bin/bash
# You should run this script in a docker container, such as:
#docker run --ipc=host --gpus '"device=0"' -it -v $(pwd):/sonyGan -v /home/lonce/3tb/DATA_OTHERS/nsynth-train.jsonwav:/mydata lonce/sonygan:2020.12.16

#usage runtrainTEST.sh -n outname configfile

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--name) name="$2"; shift ;;
		-h|--help) echo "usage:  runtrainTEST.sh -n outname configfile"; exit 1;;
        *) configfile=$1 ;;
    esac
    shift
done

if [[ ! -z "$name" ]]; then
   namearg="-n $name"
fi


#########################################################################
D=$(date +%Y.%m.%d_%H.%M.%S) #'%(%Y-%m-%d)T'
WD=${PWD##*/}

logdir=logs/logs_$D
mkdir -p $logdir
echo "LISTING of /mydata: "
ls /mydata
echo "copying your config file, $1, to your log dir: $logdir"
cp $configfile $logdir  # copy the config file into the log directory 
#RUNNING TRAIN COMMAND:
python train.py --restart  ${namearg} -c $configfile -s 5000 -l 5000 > $logdir/stdout.txt 2>$logdir/stderr.txt &
echo "train.py --restart  ${namearg} -c $configfile -s 500 -l 200 > $logdir/stdout.txt 2>$logdir/stderr.txt"
echo "...To watch the stderr output, run: "
echo "    tail -f $logdir/stderr.txt "