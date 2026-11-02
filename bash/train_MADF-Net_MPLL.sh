export CUDA_VISIBLE_DEVICES=0

cd ..
python  /root/autodl-tmp/multi_phase/train_7metrics_MPLL_GY.py \
--n_gpu \
1 \
--root_path \
/root/autodl-tmp/data_prepare_MPLL/train  \
--test_path \
/root/autodl-tmp/data_prepare_MPLL/test \
--module \
networks.LiNet.LiNet \
--dataset \
Multiphase \
--crop_inference \
standard_crop_inference \
--eval_interval \
6 \
--max_epochs \
200 \
--batch_size \
6 \
--model_name \
MADFNetV6 \
--img_size \
224 \
--base_lr \
0.01 \
--output_dir /root/autodl-tmp/model_out_LiNet_MPLL-test \
--is_savenii 0 \
--tag \
"modify the loss: wce:0.2, 0.5, 4.5+dice loss， attn:no parameter; new infusion:local and global *"
