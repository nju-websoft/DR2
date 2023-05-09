train_stage2() {
  export  CUDA_VISIBLE_DEVICES=4
  python -m torch.distributed.launch  --nproc_per_node=1 \
    train_dense_encoder.py \
    train=biencoder_stage2 \
    datasets=acordar1_metadata_content_illusnip_stage2 \
    train_datasets=[ds_acordar1_split_$1_metadata_content_illusnip,ds_acordar1_split_$((($1+1)%5))_metadata_content_illusnip,ds_acordar1_split_$((($1+2)%5))_metadata_content_illusnip] \
    dev_datasets=[ds_acordar1_split_$((($1+3)%5))_metadata_content_illusnip] \
    output_dir=train/iswc23_2stage_fold$1
}

train_stage12() {
  export  CUDA_VISIBLE_DEVICES=2
  python -m torch.distributed.launch  --nproc_per_node=1 \
    --master_port=25641 \
    train_dense_encoder.py \
    train=biencoder_stage2 \
    datasets=acordar1_metadata_content_illusnip_stage2 \
    train_datasets=[ds_acordar1_split_$1_metadata_content_illusnip,ds_acordar1_split_$((($1+1)%5))_metadata_content_illusnip,ds_acordar1_split_$((($1+2)%5))_metadata_content_illusnip] \
    dev_datasets=[ds_acordar1_split_$((($1+3)%5))_metadata_content_illusnip] \
    model_file=/my_outputs/1stage/fold$1/dpr_biencoder.0 \
    output_dir=train/iswc23_12stage_fold$1
}

for fold in {0..4}
do
  train_stage1 $fold
done