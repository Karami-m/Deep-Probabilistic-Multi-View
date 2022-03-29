#!/bin/bash

nr_gpu=1
K=160
K_sh=120
dataset='yaleb'
subdir="K"$K"_Ksh"${K_sh}"/"
tag_pre=""
exp=0
tag='LR0005_BS100_nSampls40'

config="shrd_est=z,predec=convT,p_drop=0.0,var_prior0=20.,var_prior1=.25,reg_var_wgt=0.0001";


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
    --init_nets 'none'\
    --eval_methods "NMI2_ACC2_ARI2"\
    --lr .0005\
    --lr_decay 1.\
    --lr_stpsize 1000\
    --batch_size 100\
    --n_samples 40\
    --batch_size_valid 100\
    --fold 0 \
    --seed ${exp}00\
    --warmup_lr 0\
    --warmupDelay 0\
    --tf_save 'save'\
    --eval_interval 1000\
    --save_interval 500
    # --warmup 100\
#     --to_reset_optimizer
    
    
echo "Program test finished with exit code $? at: `date`"
