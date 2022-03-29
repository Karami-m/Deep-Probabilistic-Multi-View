#!/bin/bash

nr_gpu=1
K=60
K_sh=30
dataset='n-mnist'
subdir="K"$K"_Ksh"${K_sh}"/"
tag_pre=""
exp=0
tag='LR0002_BS200_fastSVM_warmup100'

config="shrd_est=z1,gate=relu,p_drop=0.2,var_prior0=500.,var_prior1=2.,postenc1=sp,postenc2=sp,postdec2=sp,reg_var_wgt=0.0001";


subfldr_dir=${subdir}${tag_pre}${config}/${tag}_exp${exp}

echo "Starting run at: `date` for config : "${config}
cd ..

python ./main.py \
    --dataset ${dataset} \
    --hpconfig ${config}\
    --logdir ${subfldr_dir} \
    --mode "train" \
    --hidden_dim $K \
    --hidden_dim_shrd $K_sh\
    --init_nets 'unif'\
    --eval_methods "class_NMI_ACC"\
    --lr .0002\
    --lr_decay 1.\
    --lr_stpsize 200\
    --batch_size 200\
    --batch_size_valid -1\
    --fold 0 \
    --seed ${exp}00\
    --warmup_lr 0\
    --warmupDelay 0\
    --warmup 100\
    --tf_save 'save'\
    --eval_interval 100\
    --fast_SVM 
#     --to_reset_optimizer
    
    
echo "Program test finished with exit code $? at: `date`"
