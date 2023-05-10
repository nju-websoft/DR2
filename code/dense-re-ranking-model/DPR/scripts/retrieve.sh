
export  CUDA_VISIBLE_DEVICES=2
retrieve() {
  python dense_retriever.py \
    model_file=/my_outputs/$2/fold$1/dpr_biencoder.0 \
    qa_dataset=query_acordar1_split$((($1+4)%5)) \
    ctx_datatsets=[dataset_ctx_metadata,dataset_ctx_content_illusnip] \
    encoded_ctx_files=[/my_embeddings/$2/fold$1/$2_fold$1_metadata_0,/my_embeddings/$2/fold$1/$2_fold$1_content_illusnip_0] \
    out_file=/my_retrieve/$2/fold$1/$2_results_fold$1.json
}

for stage in "2stage"
do
  for fold in {0..4}
  do
    retrieve $fold $stage
  done
done