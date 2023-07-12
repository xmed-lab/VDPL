ename='scgm_VDPL_Unet'

CUDA_VISIBLE_DEVICES=0 nohup python scgm_train.py \
exp_name=A_$ename test_vendor=A > logs/A_$ename.log 2>&1 & \
CUDA_VISIBLE_DEVICES=1 nohup python scgm_train.py \
exp_name=B_$ename test_vendor=B > logs/B_$ename.log 2>&1 & \
CUDA_VISIBLE_DEVICES=2 nohup python scgm_train.py \
exp_name=C_$ename test_vendor=C > logs/C_$ename.log 2>&1 & \
CUDA_VISIBLE_DEVICES=3 nohup python scgm_train.py \
exp_name=D_$ename test_vendor=D > logs/D_$ename.log 2>&1 &
