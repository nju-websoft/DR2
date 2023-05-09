# Dense Passage Retrieval

Follow the instructions of the origin repo of [DPR](https://github.com/facebookresearch/DPR) for traning, embedding generation, and retrieval.

```
git clone https://github.com/facebookresearch/DPR.git
```
We take fold0 as an example.

## Training

```
export  CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.launch  --nproc_per_node=1 \
    train_dense_encoder.py \
    train=biencoder_stage2 \
    datasets=acordar1_metadata_content_illusnip_stage2 \
    train_datasets=[ds_acordar1_split_0_metadata_content_illusnip,ds_acordar1_split_1_metadata_content_illusnip,ds_acordar1_split_2_metadata_content_illusnip] \
    dev_datasets=[ds_acordar1_split_3_metadata_content_illusnip] \
    output_dir=train/iswc23_2stage_fold0
```

## Embedding Generation

```
export  CUDA_VISIBLE_DEVICES=4

python generate_dense_embeddings.py \
    model_file=0my_outputs/2stage/fold0/dpr_biencoder.0 \
    ctx_src=dataset_ctx_metadata \
    out_file=0my_embeddings/2stage/fold0/2stage_fold0_metadata

python generate_dense_embeddings.py \
    model_file=0my_outputs/2stage/fold0/dpr_biencoder.0 \
    ctx_src=dataset_ctx_content_illusnip \
    out_file=0my_embeddings/2stage/fold0/2stage_fold0_content_illusnip
```

## Retrieval

```
export  CUDA_VISIBLE_DEVICES=2

python dense_retriever.py \
    model_file=0my_outputs/2stage/fold0/dpr_biencoder.0 \
    qa_dataset=query_acordar1_split4 \
    ctx_datatsets=[dataset_ctx_metadata,dataset_ctx_content_illusnip] \
    encoded_ctx_files=[0my_embeddings/2stage/fold0/2stage_fold0_metadata_0,0my_embeddings/2stage/fold0/2stage_fold0_content_illusnip_0] \
    out_file=0my_retrieve/2stage/fold0/2stage_results_fold0.json
```