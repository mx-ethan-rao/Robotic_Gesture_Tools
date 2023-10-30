#!/bin/bash

# $1 is the specific sample to run (e.g. S1)
# Specify a sample to run by sh run_multisample.sh S1:S3:S6

# set -ex
set -e

trap clean EXIT SIGTERM
clean(){
    # Unbind the file descriptor and delete the pipe when all is said and done
    exec 4<&-
    exec 4>&-
    rm -f mylist_$fifoname
    kill -9 -$$
}


cd $(dirname $0)
source /home/ethanrao/anaconda3/bin/activate TCAN

# define variable here
export GPUs=(0 1 2 3 4 5 6 7)
title="5Gestures_index"
validation_set=("H" "I")
tasks=("Suturing")



thread_num=$((${#GPUs[@]} / 2))

# Create a FIFO
fifoname=$(date +"%Y%m%d_%H%M%S")
mkfifo mylist_$fifoname
# Bind a file descriptor 4 to the FIFO 
exec 4<>mylist_$fifoname
# Write data to the pipeline beforehand, as many times as you want to start a child process.
for ((i=0; i < $thread_num; i++)); do
    echo $i >&4
done



for task in "${tasks[@]}"; do
    for valid in "${validation_set[@]}"; do
        read p_idx <&4
        # The & here opens a child process to execute
        {
            echo User_$valid
            gpu_idx=$((p_idx * 2))
            export CUDA_VISIBLE_DEVICES="${GPUs[$gpu_idx]},${GPUs[$((gpu_idx + 1))]}"
            echo $CUDA_VISIBLE_DEVICES
            python preprocess/preprocess.py --out /data/mingxing/$title/$task-$valid --user_for_val $valid --task $task
            bash scripts/run_train.sh ./configs/JIGSAWS/JIGSAWS_ft.yaml $task $valid /data/mingxing/$title/$task-$valid > /data/mingxing/$title/$task-$valid/br_train.log 2>&1
            python extract_frame_features.py --config ./configs/JIGSAWS/JIGSAWS_exfm.yaml --pretrain ./exp/clip_ucf/ViT-B/16/JIGSAWS/$task-$valid/last_model.pt --savedir /data/mingxing/$title/$task-$valid/visual_features
            mkdir -p /data/mingxing/$title/$task-$valid/JIGSAWS
            python ../MS-TCN2/preprocess.py --subdataset $task \
                                            --vpath /data/mingxing/JIGSAWS \
                                            --output /data/mingxing/$title/$task-$valid/JIGSAWS \
                                            --visual_feat /data/mingxing/$title/$task-$valid/visual_features
            cd ../MS-TCN2/
            cp /data/mingxing/MS_TCN_data/JIGSAWS/mapping.txt /data/mingxing/$title/$task-$valid/JIGSAWS/mapping.txt
            bash train.sh JIGSAWS .$task.LOUO.$valid /data/mingxing/$title/$task-$valid/
            echo "JIGSAWS_LOUO_$task-$valid"
            bash test_epoch.sh JIGSAWS .$task.LOUO.$valid 100 /data/mingxing/$title/$task-$valid/ > /data/mingxing/$title/$task-$valid/result.log 2>&1 
            echo $p_idx >&4
        } &
    done
done
# Use the wait command to block the current process until all child processes have finished
wait
