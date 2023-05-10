import argparse

import pytrec_eval

measures = ['ndcg_cut_5', 'ndcg_cut_10', 'map_cut_5', 'map_cut_10']

def calculate(test_file):
    qrel = {}
    with open('data/acordar/Data/qrels.txt') as rel_file:
        for line in rel_file:
            rel = line.split('\t')
            if rel[0] not in qrel:
                qrel[rel[0]] = {}
            qrel[rel[0]][rel[2]] = int(rel[3])
    # print(qrel)

    run = {}
    with open(test_file) as run_file:
        for line in run_file:
            list = line.split('\t')
            query_id = list[0]
            dataset_id = list[2]
            score = float(list[4])
            if query_id not in run:
                run[query_id] = {}
            run[query_id][dataset_id] = float(score)
    # print(run)

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, measures)
    return evaluator.evaluate(run)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--test_file", default=None, type=str, required=True, help="The test file to be evaluated.")
    args = parser.parse_args()
    metrics = calculate(args.test_file)
    mean_metric = {x: 0.0 for x in measures}
    n = 493
    for m in metrics:
        for i in metrics[m]:
            mean_metric[i] += metrics[m][i]/n
    print(args.test_file)
    print(mean_metric)
