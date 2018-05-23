"""
Type-aware Ranking
==================

Main class for type-aware entity ranking.

Usage
-----

::

  python type_aware_ranking.py --output <file> --baseline <file> --et <file> --target <file> --model <model> --lam <lambda> --entities <k>

:Authors: Darío Garigliotti, Faegheh Hasibi, Krisztian Balog
"""

# Standard imports
import os
import argparse
from enum import Enum
from pprint import pprint
import math


# -------------------------------
# Constants

# Retrieval models and parameters
STRICT_FILTER = "strict_filtering"
SOFT_FILTER = "soft_filtering"
INTERPOLATION = "interpolation"
MODELS = [STRICT_FILTER, SOFT_FILTER, INTERPOLATION]
VALID_LAMBDAS = [round(0.05 * k, 2) for k in range(0, 21)]  # [0..1] in 0.05-long steps
VALID_PHIS = [5 + k for k in range(0, 100, 5)]  # [5..100] in 5-long steps

# -------------------------------
# Helper functions and classes

def assure_path_exists(path):
    """Assures that a path exists, eventually creating all the subdirs if needed.

    :param path: path to a file
    :return: True if and only if path is not None and path existence was assure.
    """
    ret = False

    if path:
        d = os.path.expanduser(path)
        if not os.path.exists(d):
            os.makedirs(d)
        ret = True

    return ret
    

class RetrievalResults(object):
    """It stores retrieval scores for a given query."""

    def __init__(self, query_id=None, scores=None):
        self.query = query_id if query_id else ""  # to save some space
        self.scores = scores if scores else dict()

    def append(self, doc_id, score):
        """Adds a (document, score) pair to the results."""
        self.scores[doc_id] = score

    def num_docs(self):
        """Returns the number of documents in the results."""
        return len(self.scores)

    def write_trec_format(self, output, query_id, run_id, max_rank=100):
        """Outputs in TREC format the ranked results for a given query.

        :param output: output file descriptor
        :param query_id: query ID
        :param run_id: meaningful label describing the running configuration
        :param max_rank: maximum number of results to output for the query (default: 100)
        """
        rank = 1
        for doc_id, score in sorted(self.scores.items(), key=lambda item: (item[1],  item[0]), reverse=True):
            if rank <= max_rank:
                output.write("{}\tQ0\t{}\t{}\t{}\t{}\n".format(query_id, doc_id, rank, score, run_id))
            rank += 1


class TrecRun(object):
    def __init__(self, fpath, normalize=False, remap_by_exp=False, top_k_docs=None):
        """It represents a TREC runfile.

        :param fpath:  path to a ranking file in TREC format.
        :param normalize: if True if,  normalize the scores across all retrieved documents per query.
        :param remap_by_exp: if True if, takes exp of score (e.g., when scores are negative in log domain).
        :param top_k_docs: integer for loading only top k docs per query. Default: None, i.e., load all.
        """
        self.results = dict()  # {query_id: RetrievalResults object}
        self.sum_scores = dict()  # {query_id: sum of the scores for all (doc, score) pairs}

        # Loads the file
        with open(fpath) as fin:
            for line in [l for l in (line.strip() for line in fin) if l]:
                cols = line.split()
                query_id, doc_id, score = cols[0], cols[2], float(cols[4])

                if query_id not in self.results:
                    self.results[query_id] = RetrievalResults()
                else:
                    if top_k_docs and self.results[query_id].num_docs() >= top_k_docs:
                        continue  # not break: we need to see what about other queries in runfile


                if remap_by_exp:
                    score = math.exp(score)
                self.results[query_id].append(doc_id, score)

        if normalize:
            self.normalize()

    def normalize(self):
        """For each query, it normalizes all document scores s.t. they sum up to 1, i.e. they define
        a probability distribution.
        """
        # For each query, set sum_scores (attribute explanation in __init__ method)
        for query_id, rr in self.results.items():
            self.sum_scores[query_id] = sum(map(lambda pair: pair[1], rr.scores.items()))

        query_id_set = self.results.keys()  # new var (i.e. copy), since the for loop will modify self.results
        for query_id in query_id_set:
            normalized_result = RetrievalResults()
            for doc_id, score in self.results[query_id].scores.items():
                normalized_result.append(doc_id, score / self.sum_scores[query_id])
            self.results[query_id] = normalized_result

    def lookup(self, query_id):
        """Returns a {docID: score} dictionary."""
        return self.results.get(query_id, RetrievalResults()).scores

    def __get_sum_scores(self, query_id):
        """Returns the sum of all the document scores for a given query."""
        return self.sum_scores.get(query_id, 1)


# -------------------------------
# Main class

class TypeAwareRanking(object):
    """Main class for type-aware entity ranking."""

    def __init__(self):
        self.queries = None  # dict {query_id: query}
        self.baseline = None  # TrecRun object
        self.entity_types = None  # dict {entity: {type1, type2, ...}}
        self.target_types = None  # TrecRun object

        # global cached variables
        self.__all_theta_e = {} # dict {e: {t: P(t|theta_e)}}
        self.__p_t = {}  # dict {t: P(t)}
        self.__p_t_num = {}  # KB: number of entities with type t
        self.__p_t_denom = None
        self.__mu = None

        # Per query cached variables
        self.__KLs = None  # dict{entity: KL(theta_q || theta_e)}
        self.__max_KL = None
        self.__z = None

    def load_queries(self, fpath):
        """Loads the queries from file.

        :param fpath: absolute path to the queries file (a TSV file for which each line is of the shape
        <queryID>\t<query>).
        """
        print("-" * 79 + "\nLoading queries from:\t", fpath)

        if not self.queries:
            self.queries = dict()
            with open(fpath) as fin:
                for line in [l for l in (line.strip() for line in fin) if l]:
                    query_id, query = line.split("\t")
                    self.queries[query_id] = query

    def load_baseline_ranking(self, fpath, top_k_docs=None, remap_by_exp=False):
        """Loads a term-based baseline entity ranking from file.

        :param fpath: absolute path to the entity ranking file in TREC format.
        :param top_k_docs: integer for loading only top k docs per query. Default: None, i.e., load all.
        :param remap_by_exp: if True if, takes exp of score (e.g., when scores are negative in log domain).
        """
        print("Loading baseline from:\t", fpath)

        if not self.baseline:
            self.baseline = TrecRun(fpath, normalize=True, remap_by_exp=remap_by_exp, top_k_docs=top_k_docs)

    def load_entity_types(self, fpath):
        """Loads an entity-to-type assignment from file.

        :param fpath: absolute path to the entity-to-type assignment file <entityID>\t<rdfs:isOfType>\t<typeID>).
        """
        print("Loading entity types from:\t", fpath)

        if not self.entity_types:  # to avoid overwriting a previous loading
            self.entity_types = dict()

            with open(fpath) as fin:
                for line in [l for l in (line.strip() for line in fin) if l]:
                    entity, _, t = line.split("\t")
                    if entity not in self.entity_types:
                        self.entity_types[entity] = set()
                    self.entity_types[entity].add(t)
                    self.__p_t_num[t] = self.__p_t_num.get(t, 0) + 1


    def load_target_types(self, fpath, top_k_types=None):
        """Loads a target types ranking from file.

        :param fpath: absolute path to the target types ranking file in TREC format.
        :param top_k_types: integer for loading only top k types per query (e.g., for strict filtering).
        Default: None, i.e., load all.
        """
        print("Loading target types from:\t", fpath)

        if not self.target_types:  # to avoid overwriting a previous loading
            self.target_types = TrecRun(fpath, normalize=True, top_k_docs=top_k_types)

    def rank(self, output_file, model=INTERPOLATION, lambd=0, runid="TypeAware"):
        """Creates an entity ranking using a retrieval model, and outputs it to a file.

        :param output_file: absolute path to the destination output ranking file.
        :param model: Retrieval model. Optional; Model.INTERPOLATION by default.
        :param lambd: Weighting lambda parameter in interval [0..1] when the retrieval model is interpolation.
        Optional; 0 by default.
        :param runid: Label for identifying a running configuration (i.e., combination of settings for target type,
        type taxonomy,type representation, retrieval model -and possibly the interpolation lambda-).
        """
        output = open(output_file, "w")
        for qid in sorted(self.queries):
            results = RetrievalResults()
            self.__KLs, self.__max_KL, self.__z = {}, None, None
            for e, p_qw_e in sorted(self.baseline.lookup(qid).items(), key=lambda item: (item[1], item[0]), reverse=True):
                print("Ranking", qid, e, "...")
                p_qt_e = self.get_type_sim(qid, e, model)
                if model == INTERPOLATION:
                    # P(q|e) = (1 − lambd_t)P(q_w|e) + lambd_t P(q_t|e)
                    p_q_e = ((1 - lambd) * p_qw_e) + (lambd * p_qt_e)
                else:
                    # P(q|e) = P(q_w|e) * P(q_t|e)
                    p_q_e = p_qw_e * p_qt_e # if p_qt_e !=0 else p_qw_e

#                 p_q_e = math.log(p_q_e)
                results.append(e, p_q_e)

            # NOTE: be sure that you output same max_rank as for (all) your baseline(s)
            results.write_trec_format(output, qid, runid, max_rank=1000)
        output.close()
        print("Runfile:\t{}\n".format(output_file))

    def get_type_sim(self, qid, e, model):
        """Computes type-based similarity for a given query and entity
        
        P(q_t|e) = z . (max KL(theta_q || theta_e') − KL(theta_q || theta_e)) [OR exp(- sum_t p(t|theta_q).log(P(t|theta_e)))]
        KL(theta_q || theta_e) = Sum_t P(t|theta_q) . log (P(t|theta_q) / P(t|theta_e))
        P(t|theta_e) = (n(t,e) + \mu P(t)) / Sum_t' n(t', e) + \mu
        P(t) = Sum_e n(t, e) / Sum_t Sum_e n(t, e)
        P(t|theta_q): from target types (oracle or automatic)
        """
        if model == STRICT_FILTER:
            # P(q_t|e) is 1 if the target types and entity types have a non-empty intersection, and is 0 otherwise.
            target_types = set(self.target_types.lookup(qid).keys())
            entity_types = self.entity_types.get(e, set())
            p_qt_e = 1 if len(target_types & entity_types) > 0 else 0

        else:
            # P(q_t|e) = z(max KL(theta_q || theta_e')− KL(theta_q || theta_e))
            if len(self.__KLs) == 0:
                self.compute_query_KLs(qid)
            p_qt_e = self.__z * (self.__max_KL - self.__KLs[e]) if self.__z != 0 else 1
            # z=0 can only happen iff all query entities have the exact same distribution.
            # In that case P(q_t|e) can be set to any >0 value as it won't affect the ranking

        return p_qt_e

    def compute_query_KLs(self, qid):
        """
        Computes the followings for a given query:
         - KL(theta_q || theta_e) for all entities of a query
         - max_e' KL(theta_q || theta_e')
         - z = 1 / sum_e (max_e' KL(theta_q || theta_e') − KL(theta_q || theta_e))

        ---------------
        Done only once for a query for efficiency reasons.
        """
        entities = self.baseline.lookup(qid)

        # KL(theta_q || theta_e) for all entities of a query
        theta_q = self.target_types.lookup(qid)  # theta_q: {t: P(t|theta_q)}
        for e in entities:
            theta_e = self.get_theta_e(e, qid)  # theta_e: {t: P(t|theta_e)}
            self.__KLs[e] = self.kl_divergence(theta_q,  theta_e)

        # computes max_e' KL(theta_q || theta_e')
        self.__max_KL = max(self.__KLs.values()) + 0.00001

        # computes z = 1 / sum_e (max_e' KL(theta_q || theta_e') − KL(theta_q || theta_e))
        denominator = sum([self.__max_KL - self.__KLs[e] for e in entities])
        self.__z = 1 / denominator if denominator != 0 else 0

    def get_theta_e(self, e, qid):
        """ Return a dictionary {t: P(t|theta_e)}
        P(t|theta_e) = (n(t,e)+ \mu P(t)) / Sum_t' n(t', e) + \mu
        """
        # mu = average number of types per entity
        if not self.__mu:
            self.__mu = sum([len(types) for types in self.entity_types.values()]) / len(self.entity_types.keys())

        # P(t|theta_e) = (n(t,e)+ \mu P(t)) / Sum_t' n(t', e) + \mu
        if e not in self.__all_theta_e:
            self.__all_theta_e[e] = {}
            # all_types = {t for types in self.entity_types.values() for t in types}
            for t in self.target_types.lookup(qid): #all_types:
                n_t_e = 1 if t in self.entity_types.get(e, set()) else 0
                p_t = self.get_p_t(t)
                sum_n_t_e = len(self.entity_types.get(e, set()))
                self.__all_theta_e[e][t] = (n_t_e + self.__mu * p_t) / (sum_n_t_e + self.__mu)

        return self.__all_theta_e[e]

    def get_p_t(self, t):
        """Computes background model for a given type t
        P(t) = Sum_e n(t, e) / Sum_t Sum_e n(t, e)
        """
        if t not in self.__p_t:
            # denominator:  Sum_t Sum_e n(t, e)
            if not self.__p_t_denom:
                self.__p_t_denom = sum([len(types) for types in self.entity_types.values()])

            # numerator: Sum_e n(t, e)
            # p_t_num = sum([1 for e, types in self.entity_types.items() if t in types])

            self.__p_t[t] = self.__p_t_num.get(t, 0) / self.__p_t_denom

        return self.__p_t[t]

    def kl_divergence(self, p, q):
        """Computes the KL-divergence of two probability distributions, KL(p||q).
        KL(p||q) = H(p,q) - H(p) = Sum_i p(i) * log(p(i)/q(i))

        :param p: probability distribution p {i: p(i)}
        :param q: probability distribution q {i: q(i)}
        """
        kl_divergence = 0

        for i in set(p.keys()):  # efficiency trick vs the original "for all events in universe = dom(P) union dom(Q)"
            try:
                log_arg = p.get(i, 0) / q.get(i, 0)
                kl_divergence_i = p.get(i, 0) * math.log(log_arg)
            except:
                continue
            kl_divergence += kl_divergence_i

        return kl_divergence


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output",   required=True, help="output ranking file", type=str)
    parser.add_argument("-q", "--queries",  required=True, help="queries ranking file", type=str)
    parser.add_argument("-b", "--baseline", required=True, help="baseline ranking file", type=str)
    parser.add_argument("-t", "--target",   required=True, help="target type ranking file", type=str)
    parser.add_argument("-e", "--et",       required=True, help="entity-to-type assignment file", type=str)
    parser.add_argument("-m", "--model",                   help="retrieval model", type=str, default=INTERPOLATION)
    parser.add_argument("-l", "--lam",                     help="interpolation lambda", type=float, default=0)
    parser.add_argument("-p", "--phi",                     help="strict filtering phi", type=int, default=None)
    parser.add_argument("-k", "--entities",                help="top k entities per query to load from baseline", type=int, default=None)
    parser.add_argument("-r", "--remap",                   help="flag: True iff remap into exp() space of scores is required", type=bool, default=True)

    args = parser.parse_args()

    # Parameters validation
    if not assure_path_exists(os.path.dirname(args.output)):
        raise ValueError("Invalid output parent dir: {}".format(args.output))
    if not args.model in MODELS:
        raise ValueError("Invalid model: {}".format(args.model))
    if args.model == INTERPOLATION and not args.lam in VALID_LAMBDAS:
        raise ValueError("Invalid interpolation lambda: {}".format(args.lam))
    if args.model == STRICT_FILTER and not args.phi in VALID_PHIS:
        raise ValueError("Invalid strict filtering phi: {}".format(args.phi))
    if not args.remap in [True, False]:
        raise ValueError("Invalid Boolean remap: {}".format(args.remap))

    return args


def main(args):
    """Main function."""
    ta_ranking = TypeAwareRanking()
    ta_ranking.load_queries(args.queries)
    ta_ranking.load_baseline_ranking(args.baseline, args.entities, remap_by_exp=args.remap)
    ta_ranking.load_target_types(args.target, top_k_types=args.phi)
    ta_ranking.load_entity_types(args.et)

    print("\tRanking entities...\n\t\tModel: {}\n{}{}".
          format(args.model,
                 "\t\tLambda: {}\n".format(args.lam) if args.model == INTERPOLATION else "",
                 "\t\Phi: {}\n".format(args.phi) if args.model == STRICT_FILTER else ""))
    ta_ranking.rank(args.output, model=args.model, lambd=args.lam)


if __name__ == '__main__':
    main(arg_parser())

