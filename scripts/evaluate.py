from itertools import combinations, product
from typing import List, Dict, Any, Tuple, Iterable, Set
from common import Role, Argument
from paraphrases import get_paraphrase_score
import networkx as nx
from networkx.algorithms.matching import max_weight_matching

MATCH_IOU_THRESHOLD = 0.5


class Metrics:
    def __init__(self, true_positive: int, false_positive: int, false_negative: int):
        self.true_positive = true_positive
        self.false_positive = false_positive
        self.false_negative = false_negative

    def prec(self):
        n_predicted = self.true_positive + self.false_positive
        return float(self.true_positive)/n_predicted

    def recall(self):
        n_true = self.true_positive + self.false_negative
        return float(self.true_positive)/n_true

    def f1(self):
        p, r = self.prec(), self.recall()
        return 2*p*r/(p+r)

    def __str__(self):
        return f"{self.prec()*100:5.2f}%\t{self.recall()*100:5.2f}%\t{self.f1()*100:5.2f}%"

    def as_tuple(self):
        return (self.true_positive, self.false_positive, self.false_negative)


def iou(arg1: Argument, arg2: Argument):
    joint = joint_len(arg1, arg2)
    len1 = arg1[1] - arg1[0]
    len2 = arg2[1] - arg2[0]
    union = len1 + len2 - joint
    return float(joint)/union


def joint_len(arg1: Argument, arg2: Argument):
    max_start = max(arg1[0], arg2[0])
    min_end = min(arg1[1], arg2[1])
    joint = max(min_end - max_start, 0)
    return joint


def get_overlap_arguments(grt_items, sys_items, scoring_fn, threshold):
    # Sorting is important to make comparison invariant
    # to order of iteration for the greedy matcher
    sys_items = sorted(sys_items)
    grt_items = sorted(grt_items)
    all_pairs = ((sys_item, grt_item, scoring_fn(sys_item, grt_item))
                 for sys_item, grt_item in product(sys_items, grt_items))

    matches = [(sys_item, grt_item, score) for (sys_item, grt_item, score)
               in all_pairs if score >= threshold]
    return matches


def align_one_to_one(matches: List[Tuple[Any, Any, float]]) -> Dict[Any, Any]:
    bipartite = nx.Graph()
    if not matches:
        return {}

    for sys_arg, grt_arg, score in matches:
        sys_arg = f"sys_{sys_arg[0]}:{sys_arg[1]}"
        grt_arg = f"grt_{grt_arg[0]}:{grt_arg[1]}"
        bipartite.add_edge(sys_arg, grt_arg, weight=score)
    max_alignment = max_weight_matching(bipartite, maxcardinality=True)
    sys_to_grt = {}
    for arg_1, arg_2 in max_alignment:
        sys_arg, grt_arg = (arg_1, arg_2) if "sys" in arg_1 else (arg_2, arg_1)
        sys_arg = tuple(int(coord) for coord in sys_arg[4:].split(":"))
        grt_arg = tuple(int(coord) for coord in grt_arg[4:].split(":"))
        sys_to_grt[sys_arg] = grt_arg
    return sys_to_grt


def consolidate_by_overlap(args: Iterable[Argument], scoring_fn, threshold):
    g = nx.Graph()
    g.add_nodes_from(args)
    g.add_edges_from((arg_1, arg_2)
                     for arg_1, arg_2 in combinations(args, r=2)
                     if scoring_fn(arg_1, arg_2) >= threshold)

    components = nx.connected_components(g)
    representatives = [next(iter(component)) for component in components]
    return representatives


def match_arguments(grt_args: Set[Argument],
                    sys_args: Set[Argument]):
    matches = get_overlap_arguments(grt_args, sys_args, iou, MATCH_IOU_THRESHOLD )
    sys_to_grt_arg = align_one_to_one(matches)

    matched_sys_args = set(m[0] for m in matches)
    unmatched_sys_args = sys_args - matched_sys_args
    unmatched_grt_args = set(sys_to_grt_arg.values()) - set(grt_args)
    # This extension is used to evaluate redundant datasets
    unmatched_sys_args_consolidated = consolidate_by_overlap(unmatched_sys_args, iou, MATCH_IOU_THRESHOLD )
    return sys_to_grt_arg, unmatched_sys_args_consolidated, unmatched_grt_args


def evaluate(sys_roles: List[Role],
             grt_roles: List[Role]):

    # remove duplicates from unlabelled and labeled arguments
    sys_args = set(arg for role in sys_roles for arg in role.arguments)
    grt_args = set(arg for role in grt_roles for arg in role.arguments)
    # get arguments with high overlap
    sys_to_grt_arg, unmatched_sys_args, unmatched_grt_args = match_arguments(grt_args, sys_args)

    n_unlabel_tp = len(sys_to_grt_arg)
    n_unlabel_fp = len(unmatched_sys_args)
    n_unlabel_fn = len(grt_args) - n_unlabel_tp
    unlabelled_arg_metrics = Metrics(n_unlabel_tp, n_unlabel_fp, n_unlabel_fn)

    sys_qna = set((role.question, arg) for role in sys_roles for arg in role.arguments)
    grt_qna = set((role.question, arg) for role in grt_roles for arg in role.arguments)
    n_label_tp, n_label_fp, n_label_fn = n_unlabel_tp, n_unlabel_fp, n_unlabel_fn
    matched_grt_roles = set()
    for sys_arg, grt_arg in sys_to_grt_arg.items():
        sys_q = next(q for (q, a) in sys_qna if a == sys_arg)
        grt_q = next(q for q, a in grt_qna if a == grt_arg)

        matched_grt_roles.add(grt_q)
        if not get_paraphrase_score(sys_q, grt_q):
            n_label_tp -= 1
            n_label_fp += 1
            n_label_fn += 1
    labeled_arg_metrics = Metrics(n_label_tp, n_label_fp, n_label_fn)

    n_unlabel_role_tp = len(matched_grt_roles)
    n_unlabel_role_fn = len(grt_roles) - n_unlabel_role_tp
    role_metrics = Metrics(n_unlabel_role_tp, 0, n_unlabel_role_fn)
    return unlabelled_arg_metrics, labeled_arg_metrics, role_metrics
