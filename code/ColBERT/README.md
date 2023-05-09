# ColBERT

Follow the instructions of the origin repo of [ColBERT](https://github.com/stanford-futuredata/ColBERT) for traning and testing.


```
git clone https://github.com/stanford-futuredata/ColBERT.git
```

We take fold0 as an example.

## Training

```
CUDA_VISIBLE_DEVICES=0 python -m colbert.train --amp \
        --checkpoint experiments/iswc23/acordar1/pretrain/train.py/best/checkpoints/colbert-0.dnn \
        --doc_maxlen 512 \
        --query_maxlen 64 \
        --mask-punctuation \
        --bsize 16 \
        --triples docs/acordar1/all/pairs_012.jsonl \
        --root experiments/iswc23/acordar1/all_illusnip/0/ \
        --collection docs/acordar1/illusnip/collections_all.tsv \
        --queries docs/acordar1/queries_012.tsv \
        --valid_triples docs/acordar1/all/pairs_3.jsonl \
        --valid_queries docs/acordar1/queries_3.tsv \
        --experiment 2stage \
        --similarity cosine \
        --run 1 \
        --epoch 1 \
        --lr 1e-6
```

## Test

```
CUDA_VISIBLE_DEVICES=0 python -m colbert.test --amp --doc_maxlen 512 --mask-punctuation \
    --bsize 256 \
    --collection docs/acordar1/illusnip/collections_all.tsv \
    --queries docs/acordar1/queries_4.tsv \
    --topk docs/iswc23/acordar1/all_illusnip/top10_all_BM25.tsv\
    --checkpoint experiments/iswc23/acordar1/all_illusnip/0/2stage/train.py/best/checkpoints/colbert-0.dnn \
    --root experiments/iswc23/acordar1/all_illusnip/0 \
    --experiment 2stage \
    --run BM25_top10_rerank
```