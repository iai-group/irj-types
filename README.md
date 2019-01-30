# Identifying and Exploiting Target Entity Type Information for Ad Hoc Entity Retrieval

This repository provides resources developed within the following article:

> D. Garigliotti, F. Hasibi, and K. Balog. Identifying and Exploiting Target Entity Type Information for Ad Hoc Entity Retrieval. Information Retrieval Journal, (), 1-39. [DOI: 10.1007/s10791-018-9346-x](http://link.springer.com/article/10.1007/s10791-018-9346-x)


### Abstract

> *Today, the practice of returning entities from a knowledge base in response to search queries has become widespread.  One of the distinctive characteristics of entities is that they are typed, i.e., assigned to some hierarchically organized type system (type taxonomy).  The primary objective of this paper is to gain a better understanding of how entity type information can be utilized in entity retrieval.  We perform this investigation in two settings: firstly, in an idealized ``oracle'' setting, assuming that we know the distribution of target types of the relevant entities for a given query; and secondly, in a realistic scenario, where target entity types are identified automatically based on the keyword query.  We perform a thorough analysis of three main aspects: (i) the choice of type taxonomy, (ii) the representation of hierarchical type information, and (iii) the combination of type-based and term-based similarity in the retrieval model.  Using a standard entity search test collection based on DBpedia, we show that type information can significantly and substantially improve retrieval performance, \response{yielding up to 67\% relative improvement in terms of NDCG@10 over a strong text-only baseline in an oracle setting.  We further show that using automatic target type detection, we can outperform the text-only baseline by 44\% in terms of NDCG@10.  This is as good as, and sometimes even better than, what is attainable by using explicit target type information provided by humans.  These results indicate that identifying target entity types of queries is challenging even for humans and attests to the effectiveness of our proposed automatic approach.* 

## Repository Files

This repository is structured as follows:

 - `evaluation/`: all data needed to reproduce the evaluation results presented in the article.

 - `code/type_aware_ranking.py`: a Python file with the code needed for obtaining an entity ranking, from a term-based baseline and entity type information, for a particular retrieval configuration.

 - `tti-dbpedia-ltr_features/features.json`: a JSON file containing the training set with all precomputed features for our Learning-to-Rank Target Type Identification approach.


### Entity Retrieval results

Entity Retrieval results presented in the article can be obtained as explained below.

Make sure you are in `entity_retrieval/` subdirectory:

```
$ pwd
/path/to/repo/irj-types/evaluation/entity_retrieval
```

#### Table 6

These results correspond to evaluate an entity retrieval runfile under the (uncompressed) directory `runs/type_aware/tables_6_and_7/` versus the default qrels `data/qrels/qrels.txt`:

```
$ /path/to/trec_eval -c -m ndcg_cut.10,100 data/qrels/qrels.txt runs/baseline/sdm.txt  # for the baseline
$ /path/to/trec_eval -c -m ndcg_cut.10,100 data/qrels/qrels.txt runs/type_aware/tables_6_and_7/<model>/<representation>/<taxonomy>[_<interpolation_lambda>].txt  # for type-aware
```

#### Table 7

Similarly to Table 6, but here a runfile is evaluated against the taxonomy-specific qrels `data/qrels/qrels_<taxonomy>.txt`:

```
$ /path/to/trec_eval -c -m ndcg_cut.10,100 data/qrels/qrels_<taxonomy>.txt runs/baseline/sdm.txt  # for the baseline
$ /path/to/trec_eval -c -m ndcg_cut.10,100 data/qrels/qrels_<taxonomy>.txt runs/type_aware/tables_6_and_7/<model>/<representation>/<taxonomy>[_<interpolation_lambda>].txt  # for type-aware
```

#### Table 12

These results correspond to evaluate an entity retrieval runfile under the (uncompressed) directory `runs/type_aware/table_12/` versus the DBpedia-specific qrels `data/qrels/qrels_dbpedia.txt`.
  - The subdirectory `<TTI>/` corresponds to the Target Type Identification (TTI) model used for automatically identifying target types. `<TTI>` can be: `ec/bm25/k_10`, `tc/lm`, or `ltr`.

```
$ /path/to/trec_eval -c -m ndcg_cut.10,100 data/qrels/qrels_dbpedia.txt runs/baseline/sdm.txt  # for the baseline
$ /path/to/trec_eval -c -m ndcg_cut.10,100 data/qrels/qrels_dbpedia.txt runs/type_aware/table_12/auto_types/<TTI>/<model>/<representation>/dbpedia[_<interpolation_lambda>|_<strict_filtering_k>].txt  # for automatically identified types
$ /path/to/trec_eval -c -m ndcg_cut.10,100 data/qrels/qrels_dbpedia.txt runs/type_aware/table_12/oracle2_types/<model>/<representation>/dbpedia[_<interpolation_lambda>].txt  # for Oracle 2
$ /path/to/trec_eval -c -m ndcg_cut.10,100 data/qrels/qrels_dbpedia.txt runs/type_aware/table_12/oracle1_types/<model>/<representation>/dbpedia[_<interpolation_lambda>].txt  # for Oracle 1
```


### Target Type Identification results

Target Type Identification (TTI) results presented in Table 7 can be obtained as follows.

Make sure you are now in `tti/` subdirectory and run `trec_eval` for a TTI runfile under `runs-table_10/` versus the DBpedia type ranking qrels (from [SIGIR'17 test collection](https://github.com/iai-group/sigir2017-query_types/blob/master/data/qrels/qrels-tti-CF-filtered_by_NIL%2Bmerged.tsv)):

```
$ pwd
/path/to/repo/irj-types/evaluation/tti
$ /path/to/trec_eval -c -m ndcg_cut.1,5 data/qrels/qrels-tti-dbpedia.txt runs-table_10/<TTI_model>.txt
```


## Citation

If you use the resources presented in this repository, please cite:

```
@inproceedings{Garigliotti:2018:IET,
 author =    {Dar\'{i}o Garigliotti and Faegheh Hasibi and Krisztian Balog},
 title =     {Identifying and Exploiting Target Entity Type Information for Ad Hoc Entity Retrieval},
 booktitle = {Information Retrieval Journal},
 year =      {2018},
 doi =       {10.1007/s10791-018-9346-x},
 publisher = {Springer}
}
```


## Contact

Should you have any questions, please contact Darío Garigliotti at dario.garigliotti[AT]uis.no (with [AT] replaced by @).
