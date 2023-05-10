train_stage1() {
  export CUDA_VISIBLE_DEVICES=2
  python -m torch.distributed.launch  --nproc_per_node=1 \
    --master_port=25641 \
    train_dense_encoder.py \
    train=biencoder_stage1 \
    datasets=acordar1_metadata_content_illusnip_stage1 \
    train_datasets=[ds_train_acordar1_fold_$1_metadata_content_illusnip] \
    dev_datasets=[ds_dev_acordar1_fold_$1_metadata_content_illusnip] \
    output_dir=train/iswc23_1stage_fold$1_bs_2_lr_4
}

train_stage1

for fold in {0..4}
do
  train_stage1 $fold
done