#!/bin/bash
# You should run this script in a docker container, such as:
#docker run --ipc=host --gpus '"device=0"' -it -v $(pwd):/sonyGan -v /home/lonce/3tb/DATA_OTHERS/nsynth-train.jsonwav:/mydata lonce/sonygan:2020.12.16

if [ "$#" -ne 3 ]; then
    echo "usage: keeprun from_path iteration (e.g. s4_i299940) out_dir"
    exit 1
fi

pathname=`dirname "$1"`
keepname=`basename "$1"`
origdir=${pathname}/${keepname}

iteration=$2
keepdir=$3/${keepname}

#mkdir -p ${pathname}/${keepname}_keep
mkdir -p ${keepdir}



# The configuration file, foo/foo_config.json
cp ${origdir}/${keepname}_config.json ${keepdir}
# the last checkpoint ( foo/foo_s4_i299940.pt )
cp ${origdir}/${keepname}_${iteration}.pt ${keepdir}
# the last temporary configuration of the net (e.g. foo/foo_s4_i299940_tmp_config.json)
cp ${origdir}/${keepname}_${iteration}_tmp_config.json ${keepdir}
# the training configuration file (e.g. foo/foo_train_config.json)
cp ${origdir}/${keepname}_train_config.json ${keepdir}
# error log pickle file
cp ${origdir}/${keepname}_losses.pkl ${keepdir}

mv ${origdir} ${origdir}_original

ls ${keepdir}

#########################################################################
