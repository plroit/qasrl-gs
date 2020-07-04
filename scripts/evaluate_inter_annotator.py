from argparse import ArgumentParser
from typing import List, Dict

import pandas as pd
import numpy as np
import os
from glob import glob
from itertools import combinations, product

from common import Role, Argument
from evaluate import Metrics, joint_len, iou
from evaluate_dataset import eval_datasets, yield_paired_predicates
from decode_encode_answers import decode_qasrl


def is_argument_match(arguments1: List[Argument], arguments2: List[Argument]):
    for arg1, arg2 in product(arguments1, arguments2):
        if iou(arg1, arg2) >= 0.3:
            return True
    return False


def evaluate_agreement(roles1: List[Role], roles2: List[Role]) -> int:
    used_roles1 = set()
    used_roles2 = set()
    n_matches = 0
    for role1, role2 in product(roles1, roles2):
        # if role1 in used_roles1 or role2 in used_roles2:
            # continue

        q1 = role1.question.wh.lower()
        q2 = role2.question.wh.lower()
        q1 = 'whowhat' if q1 in ('who', 'what') else q1
        q2 = 'whowhat' if q2 in ('who', 'what') else q2
        is_wh_match = q1 == q2
        if is_argument_match(role1.arguments, role2.arguments):
            if not is_wh_match:
                print(role1.question.text, role2.question.text)
        if is_wh_match and is_argument_match(role1.arguments, role2.arguments):
            n_matches += 1
            used_roles1.add(role1)
            used_roles2.add(role2)
    return n_matches


def eval_datasets_for_agreement(df1, df2):
    n_matches = 0
    n_total_roles = 0
    for key, roles1, roles2 in yield_paired_predicates(df1, df2):
        local_n_matches = evaluate_agreement(roles1, roles2)
        n_matches += local_n_matches
        n_total_roles += len(roles1) + len(roles2) - local_n_matches
    return n_matches, n_total_roles


def evaluate_generator_agreement(annot_df: pd.DataFrame, sent_map: Dict[str, List[str]]):
    cols = ['qasrl_id', 'verb_idx']
    n_gen = annot_df.groupby(cols).worker_id.transform(pd.Series.nunique)
    workers = annot_df.worker_id.unique().tolist()
    n_workers = len(workers)
    annot_df = annot_df[n_gen == n_workers].copy()
    n_predicates = annot_df[cols].drop_duplicates().shape[0]
    print("n_workers: ", n_workers)
    print("n_predicates: ", n_predicates)
    print(f"worker_1\tworker_2\tprec\trecall\tf1")

    f1s, label_f1s = [], []
    uniq_roles_per_predicate = []
    agreed_roles_per_predicate = []
    for w1, w2 in combinations(workers, r=2):
        w1_df = annot_df[annot_df.worker_id == w1].copy()
        w2_df = annot_df[annot_df.worker_id == w2].copy()
        # n_matches, n_total = eval_datasets_for_agreement(w1_df, w2_df)
        # uniq_roles_per_predicate.append(float(n_total)/n_predicates)
        # agreed_roles_per_predicate.append(float(n_matches)/n_predicates)
        #
        #
        arg_metrics, label_arg_metrics, _ = eval_datasets(w1_df, w2_df)
        print(f"{w1}\t{w2}\t{arg_metrics.prec()}\t{arg_metrics.recall()}\t{arg_metrics.f1()}")
        print(f"{w1}\t{w2}\t{label_arg_metrics.prec()}\t{label_arg_metrics.recall()}\t{label_arg_metrics.f1()}")

        f1s.append(arg_metrics.f1())
        label_f1s.append(label_arg_metrics.f1())
    f1s = np.array(f1s)
    label_f1s = np.array(label_f1s)
    print(f1s.mean(), f1s.std())
    print(label_f1s.mean(), label_f1s.std())

    # agreed_roles_per_predicate = np.array(agreed_roles_per_predicate)
    # print(agreed_roles_per_predicate.mean(), agreed_roles_per_predicate.std())
    #
    # uniq_roles_per_predicate = np.array(uniq_roles_per_predicate)
    # print(uniq_roles_per_predicate.mean(), uniq_roles_per_predicate.std())


def read_csv(file_path: str):
    try:
        return pd.read_csv(file_path)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="Latin-1")


def dataset_path(root_dir: str, dataset_name: str,
                 gen1: str, gen2: str, arb: str):
    slice_path = "_".join([gen1, gen2, arb])
    slice_path = f"{dataset_name}.inter.{slice_path}.csv"
    slice_path = os.path.join(root_dir, slice_path)
    return slice_path


def main(root_dir: str, dataset_name: str):
    readme = pd.read_csv(os.path.join(root_dir, 'readme.csv'))
    sent_path = os.path.join(root_dir, f'{dataset_name}.csv')
    sent_df = read_csv(sent_path)
    sent_map = dict(zip(sent_df.qasrl_id, sent_df.tokens.apply(str.split)))
    # original annotations, multiple generation tasks per predicate
    annot_df = read_csv(os.path.join(root_dir, f'{dataset_name}.annot.csv'))
    annot_df = decode_qasrl(annot_df)
    print(annot_df.worker_id.value_counts())
    evaluate_generator_agreement(annot_df, sent_map)

    slice_pairs = []
    for arbitrators_, generators_ in zip(readme.arbitrators, readme.generators):
        arb1, arb2 = arbitrators_.split()
        gen1, gen2, gen3, gen4 = generators_.split()
        slice1_path = dataset_path(root_dir, dataset_name, gen1, gen2, arb1)
        slice2_path = dataset_path(root_dir, dataset_name, gen3, gen4, arb2)
        slice1 = decode_qasrl(pd.read_csv(slice1_path))
        slice2 = decode_qasrl(pd.read_csv(slice2_path))
        # make sure they have the same predicates...
        s1 = set(zip(slice1.qasrl_id, slice1.verb_idx))
        s2 = set(zip(slice2.qasrl_id, slice2.verb_idx))
        print(len(s1), len(s2))
        unlabelled_arg, labeled_arg, unlabelled_role = eval_datasets(slice1, slice2)
        print(unlabelled_arg)
        print(labeled_arg)
        print(unlabelled_role)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("inter_annotator_dir")
    ap.add_argument("dataset_name")
    args = ap.parse_args()
    main(args.inter_annotator_dir, args.dataset_name)
