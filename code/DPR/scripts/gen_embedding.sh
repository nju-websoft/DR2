export  CUDA_VISIBLE_DEVICES=4

gen_embeddings() {
  python generate_dense_embeddings.py \
    model_file=/my_outputs/$2/fold$1/dpr_biencoder.0 \
    ctx_src=dataset_ctx_metadata \
  	out_file=/my_embeddings/$2/fold$1/$2_fold$1_metadata
  python generate_dense_embeddings.py \
    model_file=/my_outputs/$2/fold$1/dpr_biencoder.0 \
    ctx_src=dataset_ctx_content_illusnip \
    out_file=/my_embeddings/$2/fold$1/$2_fold$1_content_illusnip
}

for stage in "2stage"
do
  for fold in {0..4}
  do
    gen_embeddings $fold $stage
  done
done