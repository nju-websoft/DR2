import datasets
import pandas as pd

dataset_set = set()
data_dict = {}
metadata_dict = {}
query_dict = {}


def read_queries(query_file):
    df = pd.read_csv(query_file, sep='\t', header=None)
    for index, row in df.iterrows():
        query_id = int(row[0])
        query_text = row[1]
        query_dict[query_id] = query_text


def read_qrels(qrel_path):
    rel_pairs = {}
    for data_type in ['train', 'valid', 'test']:
        rel_pairs[data_type] = []
        qrel_file = qrel_path + data_type + '.txt'
        df = pd.read_csv(qrel_file, sep='\t', header=None)
        for index, row in df.iterrows():
            rel_score = int(row[3])
            if rel_score == 0:
                continue
            query_id = int(row[0])
            dataset_id = int(row[2])
            dataset_set.add(dataset_id)
            rel_pairs[data_type].append((query_id, dataset_id, rel_score))
    return rel_pairs


def read_metadata(metadata_file):
    df = pd.read_csv(metadata_file, sep='\t')
    for index, row in df.iterrows():
        dataset_id = int(row['id'])
        if dataset_id not in dataset_set:
            continue
        metadata = row['text']
        metadata_dict[dataset_id] = metadata


def read_data(data_file):
    df = pd.read_csv(data_file, sep='\t')
    for index, row in df.iterrows():
        dataset_id = int(row['id'])
        if dataset_id not in dataset_set:
            continue
        data = str(row['text'])
        data_dict[dataset_id] = data


def get_data(split):
    read_queries('./data/acordar/Data/all_queries.txt')
    rel_pairs = read_qrels(f'./data/acordar/Splits/fold{split}/')
    read_metadata('../data/metadata.tsv')
    # read_data('./data/data_pcsg_T5.tsv')
    read_data('./data/data_illusnip_T5.tsv')
    train_data, valid_data, test_data = [], [], []

    for qrel in rel_pairs['train']:
        train_data.append({"dataset_id": qrel[1],
                           "metadata": metadata_dict[qrel[1]],
                           "data": data_dict[qrel[1]],
                           "keyword": query_dict[qrel[0]]
                           })
    for qrel in rel_pairs['valid']:
        valid_data.append({"dataset_id": qrel[1],
                           "metadata": metadata_dict[qrel[1]],
                           "data": data_dict[qrel[1]],
                           "keyword": query_dict[qrel[0]]
                           })
    for qrel in rel_pairs['test']:
        test_data.append({"dataset_id": qrel[1],
                          "metadata": metadata_dict[qrel[1]],
                          "data": data_dict[qrel[1]],
                          "keyword": query_dict[qrel[0]]
                          })

    train_dataset = datasets.Dataset.from_list(train_data)
    valid_dataset = datasets.Dataset.from_list(valid_data)
    test_dataset = datasets.Dataset.from_list(test_data)

    # train_dataset = train_dataset.shuffle(seed=1234)

    return train_dataset, valid_dataset, test_dataset


def get_unannotated_dataset(text_field):
    qrel_file = '../data/ACORDAR/qrels.txt'
    metadata_file = '../data/metadata.tsv'
    data_file = '../data/content_illusnip.tsv'
    # data_file = '../data/collections_pcsg.tsv'
    annotated_dataset_set = set()
    df = pd.read_csv(qrel_file, sep='\t', header=None)
    for index, row in df.iterrows():
        dataset_id = int(row[2])
        annotated_dataset_set.add(dataset_id)
    unannotated_metadata = []
    if text_field == 'metadata':
        df = pd.read_csv(metadata_file, sep='\t')
    if text_field == 'data':
        df = pd.read_csv(data_file, sep='\t')
    for index, row in df.iterrows():
        dataset_id = int(row['id'])
        metadata = str(row['text'])
        if dataset_id not in annotated_dataset_set:
            unannotated_metadata.append({'dataset_id': dataset_id, text_field: metadata})
    return datasets.Dataset.from_list(unannotated_metadata)
