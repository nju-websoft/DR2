export OUTPUT_DIR=../outputs/

predict() {
  CUDA_VISIBLE_DEVICES=0 python3 ../src/generator.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=T5 \
    --model_checkpoint=/home2/qschen/Models/t5-base/ \
    --max_input_length=512 \
    --max_target_length=32 \
    --beam_search_size=4 \
    --no_repeat_ngram_size=2 \
    --do_predict \
    --warmup_proportion=0. \
    --seed=1234 \
    --text_field=data \
    --fold=$1
}

for fold in {0..4}
  do
    predict $fold
  done