ename='mms_VDPL_Unet'

CUDA_VISIBLE_DEVICES=1,3,5,7 accelerate launch --main_process_port 12346 mms_train.py \
exp_name=A_$ename test_vendor=A ratio=0.02 > logs/A_$ename.log 2>&1 & \
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 12347 mms_train.py \
# exp_name=B_$ename test_vendor=B ratio=0.02 > logs/B_$ename.log 2>&1 & \
# CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --main_process_port 12346 mms_train.py \
# exp_name=C_$ename test_vendor=C ratio=0.02 > logs/C_$ename.log 2>&1 & \
# CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --main_process_port 12348 mms_train.py \
# exp_name=D_$ename test_vendor=A ratio=0.02 > logs/D_$ename.log 2>&1 &
