# Identifying and Exploiting Target Entity Type Information for Ad Hoc Entity Retrieval

This repository provides resources developed within the following article:

> D. Garigliotti, F. Hasibi, and K. Balog. Identifying and Exploiting Target Entity Type Information for Ad Hoc Entity Retrieval. (Under review.)

This repository is structured as follows:

 - `evaluation/`: all data needed to reproduce the evaluation results presented in the article.

 - `code/type_aware_ranking.py`: a Python file with the code needed for obtaining an entity ranking, from a term-based baseline and entity type information, for a particular retrieval configuration.

 - `tti-dbpedia-ltr_features/features.json`: a JSON file containing the training set with all precomputed features for our Learning-to-Rank Target Type Identification approach.


## Entity Retrieval results

Entity Retrieval results presented in the article can be obtained as explained below.

Make sure you are in `entity_retrieval/` subdirectory:

```
$ pwd
/path/to/repo/irj-types/evaluation/entity_retrieval
```

### Table 5

These results correspond to evaluate an entity retrieval runfile under the (uncompressed) directory `runs/type_aware/tables_5_and_6/` versus the default qrels `data/qrels/qrels.txt`:

```
$ /path/to/trec_eval -c -m ndcg_cut.10,100 data/qrels/qrels.txt runs/baseline/sdm.txt  # for the baseline
$ /path/to/trec_eval -c -m ndcg_cut.10,100 data/qrels/qrels.txt runs/type_aware/tables_5_and_6/<model>/<representation>/<taxonomy>[_<interpolation_lambda>].txt  # for type-aware
```

### Table 6

Similarly to Table 5, but here a runfile is evaluated against the taxonomy-specific qrels `data/qrels/qrels_<taxonomy>.txt`:

```
$ /path/to/trec_eval -c -m ndcg_cut.10,100 data/qrels/qrels_<taxonomy>.txt runs/baseline/sdm.txt  # for the baseline
$ /path/to/trec_eval -c -m ndcg_cut.10,100 data/qrels/qrels_<taxonomy>.txt runs/type_aware/tables_5_and_6/<model>/<representation>/<taxonomy>[_<interpolation_lambda>].txt  # for type-aware
```

### Table 9

These results correspond to evaluate an entity retrieval runfile under the (uncompressed) directory `runs/type_aware/table_9/` versus the DBpedia-specific qrels `data/qrels/qrels_dbpedia.txt`.
  - The subdirectory `<TTI>/` corresponds to the Target Type Identification (TTI) model used for automatically identifying target types. `<TTI>` can be: `ec/bm25/k_10`, `tc/lm`, or `ltr`.

```
$ /path/to/trec_eval -c -m ndcg_cut.10,100 data/qrels/qrels_dbpedia.txt runs/baseline/sdm.txt  # for the baseline
$ /path/to/trec_eval -c -m ndcg_cut.10,100 data/qrels/qrels_dbpedia.txt runs/type_aware/table_9/auto_types/<TTI>/<model>/<representation>/dbpedia[_<interpolation_lambda>|_<strict_filtering_k>].txt  # for automatically identified types
$ /path/to/trec_eval -c -m ndcg_cut.10,100 data/qrels/qrels_dbpedia.txt runs/type_aware/table_9/oracle2_types/<model>/<representation>/dbpedia[_<interpolation_lambda>].txt  # for Oracle 2
$ /path/to/trec_eval -c -m ndcg_cut.10,100 data/qrels/qrels_dbpedia.txt runs/type_aware/table_9/oracle1_types/<model>/<representation>/dbpedia[_<interpolation_lambda>].txt  # for Oracle 1
```


## Target Type Identification results

Target Type Identification (TTI) results presented in Table 7 can be obtained as follows.

Make sure you are now in `tti/` subdirectory and run `trec_eval` for a TTI runfile under `runs-table_7/` versus the DBpedia type ranking qrels (from [SIGIR'17 test collection](https://github.com/iai-group/sigir2017-query_types/blob/master/data/qrels/qrels-tti-CF-filtered_by_NIL%2Bmerged.tsv)):

```
$ pwd
/path/to/repo/irj-types/evaluation/tti
$ /path/to/trec_eval -c -m ndcg_cut.1,5 data/qrels/qrels-tti-dbpedia.txt runs-table_7/<TTI_model>.txt
```
