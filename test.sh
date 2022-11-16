export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch \
--nproc_per_node=2 train.py \
--respath checkpoints/BiSeSTDC_seg/ \
--backbone BiSeSTDCNet \
--mode train \
--n_workers_train 2 \
--n_workers_val 1 \
--max_iter 160000 \
--n_img_per_gpu 6 \
--use_boundary_8 true \
--pretrain_path checkpoints/BIiSeSTDCNePre.tar

