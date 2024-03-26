# DR2
This is the source code and data with the paper "Dense Re-Ranking with Weak Supervision for RDF Dataset Search".

## Requirements

This code is based on Python 3.7+ and Pytorch 1.9+ and uses the [Hugging Face Transformers](https://github.com/huggingface/transformers) library. The following is partial list of the required packages.

- torch
- transformers
- datasets
- faiss-cpu/faiss-gpu
- pandas
- tqdm

## Compact Document Representation

We applied two RDF snippet generation methods [Illusnip](https://github.com/nju-websoft/BANDAR/blob/master/code/src/snippetAlgorithm/IlluSnip.java) and [PCSG](https://github.com/nju-websoft/PCSG) to extract representative RDF triples of datasets and concatenated the results into compact documents. You can run codes from the origin repos for triple extraction and run codes in `./code/compact-document-representation` for compact document creation.

## Dense Re-Ranking

For the dense re-ranking models, we reused the open source code of [ColBERT](https://github.com/stanford-futuredata/ColBERT) and [DPR](https://github.com/facebookresearch/DPR). You can simply download the data and modify the configuration file to train and test the dense re-ranking model. See README at `./code/dense-re-ranking-model/ColBERT` and `./code/dense-re-ranking-model/DPR` for details.

## Coarse Tuning with Weak Supervision

### Coarse-Tuning Based on Distant Supervision
The training data used for coarse-tuning based on distant supervision is from 700K+ dataset metadata crawled from open data portals. We used the title in each metadata as keyword query and masked this field in the metadata document. The keyword queries with their corresponding dataset ids are randomly split into 90% for training and 10% for validataion and are formatted as tsv files at `./data/distant-supervision/{train/valid}.tsv`.
```
dataset_id1 query1
dataset_id2 query2
...
```
The collection of metadata is at `./data/distant-supervision/collection-of-metadata.rar`. The metadata file is in json format as follows.

```
{
    [
        {
            "dataset_id": "xxx",
            "title": "xxx",
            "description": "xxx",
            "author": "xxx",
            "tags": "[\"xxx\"]",
            "url": "https://xxx.xxx"
        },
        ...
    ]
}
```

This set of labeled data is used for coarse-tuning based on distant supervision.

### Coarse-Tuning Based on Self-Training

Source code for the document-to-query generator is at `./code/document-to-query-generator/src`.

You can use following scripts at `./code/document-to-query-generator/scripts` for training and prediction. 
```
bash ./code/document-to-query-generator/scripts/train.sh
bash ./code/document-to-query-generator/scripts/predict.sh
```

The labeled data genrated for coarse-tuning based on self-training are at `./data/self-training`. The keyword queries generated by the document-to-query generator trained for each fold with each type of document are at `./data/self-training/{metdata/data_illusnip/data_pcsg}/{metdata/data_illusnip/data_pcsg}_fold{0-4}_pred.json` in json format as follows.

```
[
    {
        "dataset_id": xxx, 
        "prediction": "xxx"
    },
    ...
]
```

### Fine-Tuning

We followed 5-fold cross-validation in [ACORDAR](https://github.com/nju-websoft/ACORDAR/tree/main/Data/Splits%20for%20Cross%20Validation), using the training and validation data from each fold for fine-tuning.


## Evaluation
All the results for re-ranking experiments are at `./outputs`. The result files are named as `all_{snippet_generation_method}_{normal_retrieval_model}_{dense_model}_{tuning}_reranking.tsv` and in TREC format. The numbers of `{tuning}` indicate different training strategies, with 0 indicating coarse-tuning based on distant supervision, 1 indicating coarse-tuning based on self-training, and 2 indicating normal fine-tuning. For example, `all_illusnip_BM25_ColBERT_012stage_reranking.tsv` means the re-ranking results of ColBERT trained by normal fine-tuning adding both coarse-tuning based on distant supervision and on self-training with Illusnip as snippet generation method and BM25 as normal retrieval model.
```
105	Q0	70723	1	28.631393432617188	ColBERT-BM25
105	Q0	11080	2	28.019168853759766	ColBERT-BM25
105	Q0	11768	3	27.970335006713867	ColBERT-BM25
...
```

For reproducing the result data for Table 1, Table 2, and Table 3 of the paper, the following script can be used to calculate the experimental metrics.

```
python ./code/evaluation/eval.py \
    --test_file=./outputs/all_illusnip_BM25_ColBERT_012stage_reranking.tsv
```
## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Citation
```
@inproceedings{DR2,
  author       = {Qiaosheng Chen and
                  Zixian Huang and
                  Zhiyang Zhang and
                  Weiqing Luo and
                  Tengteng Lin and
                  Qing Shi and
                  Gong Cheng},
  title        = {Dense Re-Ranking with Weak Supervision for {RDF} Dataset Search},
  booktitle    = {The Semantic Web - {ISWC} 2023 - 22nd International Semantic Web Conference,
                  Athens, Greece, November 6-10, 2023, Proceedings, Part {I}},
  series       = {Lecture Notes in Computer Science},
  volume       = {14265},
  pages        = {23--40},
  publisher    = {Springer},
  year         = {2023},
  url          = {https://doi.org/10.1007/978-3-031-47240-4\_2},
  doi          = {10.1007/978-3-031-47240-4\_2}
}
```
