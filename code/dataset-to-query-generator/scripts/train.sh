export OUTPUT_DIR=../T5_results/

train(){
  CUDA_VISIBLE_DEVICES=2 python3 ../src/generator.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=T5 \
    --model_checkpoint=/home2/qschen/Models/t5-base/ \
    --max_input_length=512 \
    --max_target_length=32 \
    --learning_rate=1e-5 \
    --num_train_epochs=10 \
    --batch_size=16 \
    --beam_search_size=4 \
    --no_repeat_ngram_size=2 \
    --do_train \
    --warmup_proportion=0. \
    --seed=1234 \
    --text_field=data \
    --fold=$1
}

for fold in {0..4}
  train $fold $bs $lr
done