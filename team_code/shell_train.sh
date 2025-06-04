#!/bin/bash

export CARLA_ROOT=/home/czg/CARLA_0.9.10.1
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/miniconda3/lib

export OMP_NUM_THREADS=20  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
torchrun --nnodes=1 \
--nproc_per_node=2 \
--max_restarts=1 \
--rdzv_id=42353467 \
--rdzv_backend=c10d train.py \
--id two_stage_001 \
--batch_size 1 \
--setting 02_05_withheld \
--root_dir /media/czg/Temp_data/geely_data \
--logdir ../logdir \
--use_controller_input_prediction 0 \
--use_wp_gru 0 \
--use_discrete_command 0 \
--use_tp 1 \
--continue_epoch 0 \
--cpu_cores 32 \
--num_repetitions 3 \
# --load_file logdir/train_id_003/model_0030.pth
