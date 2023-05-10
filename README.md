# RDF-Dense-Re-Ranking
This is the source code of the paper "Dense Re-Ranking with Weak Supervision for RDF Dataset Search".

## Requirements

This code is based on Python 3.7+ and Pytorch 1.9+ and uses the [Hugging Face Transformers](https://github.com/huggingface/transformers) library. The following is partial list of the required packages.

- torch
- transformers
- datasets
- faiss-cpu/faiss-gpu
- pandas
- tqdm

## Compact Document Representation

We applied two RDF snippet generation methods [Illusnip](https://github.com/nju-websoft/BANDAR/blob/master/code/src/snippetAlgorithm/IlluSnip.java) and [PCSG](https://github.com/nju-websoft/PCSG) to extract representative RDF triples of datasets and concatenated the results into pseudo documents. You can run codes from the origin repos for triple extraction and run codes in `./code/compact-document-representation` for pseudo document creation.

## Dense Re-Ranking

For training the dense re-ranking models, we reused the open source code of [ColBERT](https://github.com/stanford-futuredata/ColBERT) and [DPR](https://github.com/facebookresearch/DPR). You can simply download the data and modify the configuration file to train and test the dense re-ranking model. See README at `./code/ColBERT` and `./code/DPR` for details.

## Coarse-to-Fine Tuning

### Coarse-Tuning Based on Distant Supervision
The training data used for coarse-tuning based on distant supervision is from 700K+ dataset metadata crawled from open portals. We used the title in each metadata as pseudo query and masked this field in the metadata document. The pseudo queries and collection of metadata are at `./outputs/distant-supervised-coarse-tuning`. The metadata file is in json format as following.

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

These set of pseudo query-dataset pairs are be used for coarse-tuning based on distant supervision.

### Coarse-Tuning Based on Self-Training

Source code for the dataset-to-query generator is at `./code/dataset-to-query-generator/src`. The queries generated by the generator are at `./outputs/generation_reults` in json format as following.

```
[
    {
        "dataset_id": xxx, 
        "prediction": "xxx"
    },
    ...
]
```

You can use following scripts at `./code/dataset-to-query-generator/scripts` for trainig, testing and predictoin. 
```
bash ./code/dataset-to-query-generator/scripts/trian.sh
bash ./code/dataset-to-query-generator/scripts/test.sh
bash ./code/dataset-to-query-generator/scripts/predict.sh
```

The input datasets and corresponding output queries constitute the pseudo training data for coarse-tuning based on self-training.

### Fine-Tuning

We followed 5-fold cross-validation in [ACORDAR](https://github.com/nju-websoft/ACORDAR/tree/main/Data/Splits%20for%20Cross%20Validation), using the training and validation data from each fold for fine-tuning.


## Evaluation
All the results for re-ranking experiments are at `./outputs/re-ranking_results`. The result files are named as `all_{sampler}_{retriever}_{re-ranker}_{tuning_stage}_reranking.tsv` and in TREC format. The numbers of `{tuning_stage}` indicate different training strategies, with 0 indicating coarse-tuning stage based on distant supervision, 1 indicating coarse-tuning stage based on self-training, and 2 indicating fine-tuning stage. For example, `all_illusnip_BM25_ColBERT_012stage_reranking.tsv` means the re-ranking results of ColBERT trained by all three stages with Illusnip sampler and BM25 retriever.

For reproducing the result data for Table 2, Table 3 and Table 4 of the paper, the following script can be used to calculate the experimental metrics.

```
python .\scripts\eval.py \
    --test_file=outputs\re-ranknig_results\all_illusnip_BM25_ColBERT_02stage_reranking.tsv
```
## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Citation