#!/bin/bash
#PBS -P gansynth_train
#PBS -j oe
#PBS -N tensorflow
#PBS -q volta_gpu
#PBS -l select=1:ncpus=5:mem=80gb:ngpus=1
#PBS -l walltime=72:00:00

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



# Change to directory where job was submitted
if [ x"$PBS_O_WORKDIR" != x ] ; then
 cd "$PBS_O_WORKDIR" || exit $?
fi


# might need to create the directory logs first
echo "wd is $PBS_O_WORKDIR" >>logs/batchlog.txt

# make sure data is in the fast /scratch location (because scratch on atlas8 gives faster access than hpctmp on atlas9)
# First, create space on scratch - remember, this will be on atlas8.nus.edu which you cannot log on to directly!
mkdir -p /scratch/cnmwll
mkdir -p /scratch/cnmwll/data
echo "now rsync data to scratch" >>logs/batchlog.txt

#Now copy from where I keep the dataset permanently to where this batch job will use it
rsync -rhav /hpctmp/cnmwll/nsynth-train  /scratch/cnmwll
echo "done with rsync" >>logs/batchlog.txt

ls /scratch/cnmwll >>logs/batchlog.txt

np=$(cat ${PBS_NODEFILE} | wc -l);
echo "np is $np"

image=/app1/common/singularity-img/3.0.0/user_img/freesound-gpst-pytorch_1.7_ngc-20.12-py3.simg

singularity exec $image bash <<EOF > logs/stdout.$PBS_JOBID 2> logs/stderr.$PBS_JOBID

echo "train!!!!" >>logs/batchlog.txt

#########################################################################
D=$(date +%Y.%m.%d_%H.%M.%S) #'%(%Y-%m-%d)T'
echo $D
WD=${PWD##*/}

logdir=logs/logs_$D
mkdir -p $logdir
mkdir -p $name             # need our output dir on fast scratch space, too
echo "copying your config file, $1, to your log dir: $logdir"
cp $configfile $logdir  # copy the config file into the log directory 



#RUNNING TRAIN COMMAND:
#python train.py --restart  ${namearg} -c $configfile -s 500 -l 200 > $logdir/stdout.txt 2>$logdir/stderr.txt &
#time python train.py --restart  ${namearg} -c $configfile -s 500 -l 200 > $logdir/stdout.txt 2>$logdir/stderr.txt &
echo "train.py --restart  ${namearg} -c $configfile -s 500 -l 200 > $logdir/stdout.txt 2>$logdir/stderr.txt"
echo "...To watch the stderr output, run: "
echo "    tail -f $logdir/stderr.txt "

