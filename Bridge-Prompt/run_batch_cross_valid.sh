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
validation_set=("B" "C" "D" "E" "F" "G" "H" "I")
tasks=("Knot_Tying" "Suturing" "Needle_Passing")



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
            python preprocess/preprocess.py --out /data/mingxing/tmp/$task-$valid --user_for_val $valid --task $task
            bash scripts/run_train.sh ./configs/JIGSAWS/JIGSAWS_ft.yaml $task $valid /data/mingxing/tmp/$task-$valid
            echo $p_idx >&4
        } &
    done
done
# Use the wait command to block the current process until all child processes have finished
wait
