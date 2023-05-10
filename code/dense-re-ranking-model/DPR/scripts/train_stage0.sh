train_stage0() {
  export  CUDA_VISIBLE_DEVICES=4
  python -m torch.distributed.launch  --nproc_per_node=1\
    train_dense_encoder.py \
    train=biencoder_stage0 \
    datasets=stage0 \
    train_datasets=[ds_train_acordar1_fold_0_unsupervised_data] \
    dev_datasets=[ds_dev_acordar1_fold_0_unsupervised_data] \
    output_dir=train/iswc23_0stage_bs_2_lr_2e-5
}

train_stage0