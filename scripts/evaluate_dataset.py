import os

from typing import List, Dict, Generator, Tuple
import pandas as pd
import numpy as np
from argparse import ArgumentParser

from tqdm import tqdm

from evaluate import evaluate, Metrics, match_arguments
from common import Question, Role, QUESTION_FIELDS, Argument
from decode_encode_answers import NO_RANGE, decode_qasrl


def to_arg_roles(roles: List[Role]):
    return [(arg, role.question) for role in roles for arg in role.arguments]


def build_all_arg_roles(sys_roles: List[Role],
                        grt_roles: List[Role],
                        sys_to_grt_matches: Dict[Argument, Argument]):
    grt_arg_roles = to_arg_roles(grt_roles)
    sys_arg_roles = to_arg_roles(sys_roles)
    
    grt_arg_roles = pd.DataFrame(grt_arg_roles, columns=["grt_arg", "grt_role"])
    sys_arg_roles = pd.DataFrame(sys_arg_roles, columns=["sys_arg", "sys_role"])
    # Dictionary mapping with None values
    sys_arg_roles['grt_arg'] = sys_arg_roles.sys_arg.apply(sys_to_grt_matches.get)
    all_arg_roles = pd.merge(sys_arg_roles, grt_arg_roles, on="grt_arg", how="outer")
    all_arg_roles.grt_arg.fillna(NO_RANGE, inplace=True)
    all_arg_roles.sys_arg.fillna(NO_RANGE, inplace=True)

    return all_arg_roles


def filter_ids(df, row):
    return (df.qasrl_id == row.qasrl_id) & (df.verb_idx == row.verb_idx)


def fill_answer(arg: Argument, tokens: List[str]):
    if arg == NO_RANGE:
        return NO_RANGE
    return " ".join(tokens[arg[0]: arg[1]])


def eval_datasets(grt_df, sys_df) -> Tuple[Metrics, Metrics, Metrics]:
    unlabelled_arg_counts = np.zeros(3, dtype=np.float32)
    labelled_arg_counts = np.zeros(3, dtype=np.float32)
    unlabelled_role_counts = np.zeros(3, dtype=np.float32)
    for key, sys_roles, grt_roles in yield_paired_predicates(sys_df, grt_df):
        local_arg, local_qna, local_role = evaluate(sys_roles, grt_roles)

        unlabelled_arg_counts += np.array(local_arg.as_tuple())
        labelled_arg_counts += np.array(local_qna.as_tuple())
        unlabelled_role_counts += np.array(local_role.as_tuple())

    unlabelled_arg_counts = Metrics(*unlabelled_arg_counts)
    labelled_arg_counts = Metrics(*labelled_arg_counts)
    unlabelled_role_counts = Metrics(*unlabelled_role_counts)

    return unlabelled_arg_counts, labelled_arg_counts, unlabelled_role_counts


def build_alignment(sys_df, grt_df, sent_map):
    all_matches = []
    paired_predicates = tqdm(yield_paired_predicates(sys_df, grt_df), leave=False)
    for (qasrl_id, verb_idx), sys_roles, grt_roles in paired_predicates:
        tokens = sent_map[qasrl_id]
        grt_args = set(arg for role in grt_roles for arg in role.arguments)
        sys_args = set(arg for role in sys_roles for arg in role.arguments)
        sys_to_grt_arg, unmatched_sys_args, unmatched_grt_args = match_arguments(grt_args, sys_args)

        sys_roles_consolidated = [Role(role.question,
                                       set(arg for arg in role.arguments
                                        if arg in sys_to_grt_arg or arg in unmatched_sys_args)
                                       ) for role in sys_roles]
        # TODO:
        # Our consolidation of redundancies is based only on arguments and may remove questions
        # This is more common for the parser that predicts spans and their questions independently
        sys_roles_consolidated = [role for role in sys_roles_consolidated if role.arguments]

        all_args = build_all_arg_roles(sys_roles_consolidated, grt_roles, sys_to_grt_arg)
        all_args['qasrl_id'] = qasrl_id
        all_args['verb_idx'] = verb_idx
        all_args['grt_arg_text'] = all_args.grt_arg.apply(fill_answer, tokens=tokens)
        all_args['sys_arg_text'] = all_args.sys_arg.apply(fill_answer, tokens=tokens)
        all_matches.append(all_args)

    all_matches = pd.concat(all_matches)
    all_matches = all_matches[['grt_arg_text', 'sys_arg_text',
                                   'grt_role', 'sys_role',
                                   'grt_arg', 'sys_arg',
                                   'qasrl_id', 'verb_idx']].copy()
    return all_matches


def main(proposed_path: str, reference_path: str, sents_path=None):
    sys_df = decode_qasrl(pd.read_csv(proposed_path))
    grt_df = decode_qasrl(pd.read_csv(reference_path))
    unlabelled_arg, labelled_arg, unlabelled_role = eval_datasets(grt_df, sys_df)
    print("Metrics:\tPrecision\tRecall\tF1")
    print(f"Unlabelled Argument: {unlabelled_arg}")
    print(f"labelled Argument: {labelled_arg}")
    print(f"Unlabelled Role: {unlabelled_role}")

    print("Metrics:\tTP\tFP\tFN")
    print(f"Unlabelled Argument: {' '.join(str(t) for t in unlabelled_arg.as_tuple())}")
    print(f"labelled Argument: {' '.join(str(t) for t in labelled_arg.as_tuple())}")
    print(f"Unlabelled Role: {' '.join(str(t) for t in unlabelled_role.as_tuple())}")

    if sents_path is not None:
        sents = pd.read_csv(sents_path)
        sent_map = dict(zip(sents.qasrl_id, sents.tokens.apply(str.split)))
        align = build_alignment(sys_df, grt_df, sent_map)
        b1_dir, b1_name = os.path.split(proposed_path)
        b1 = os.path.splitext(b1_name)[0]
        b2 = os.path.splitext(os.path.basename(reference_path))[0]
        align_path = os.path.join(b1_dir, f"{b1}_{b2}.align.csv")
        print(align_path)
        align.sort_values(['qasrl_id','verb_idx', 'grt_role'], inplace=True)
        align.to_csv(align_path, encoding="utf-8", index=False)


def yield_paired_predicates(sys_df: pd.DataFrame, grt_df: pd.DataFrame):
    predicate_ids = grt_df[['qasrl_id', 'verb_idx']].drop_duplicates()
    for idx, row in predicate_ids.iterrows():
        sys_arg_roles = sys_df[filter_ids(sys_df, row)].copy()
        grt_arg_roles = grt_df[filter_ids(grt_df, row)].copy()
        sys_roles = list(yield_roles(sys_arg_roles))
        grt_roles = list(yield_roles(grt_arg_roles))
        yield (row.qasrl_id, row.verb_idx), sys_roles, grt_roles


def question_from_row(row: pd.Series) -> Question:
        question_as_dict = {question_field: row[question_field]
                            for question_field in QUESTION_FIELDS}
        question_as_dict['text'] = row.question
        return Question(**question_as_dict)


def yield_roles(predicate_df: pd.DataFrame) -> Generator[Role, None, None]:
    for row_idx, role_row in predicate_df.iterrows():
        question = question_from_row(role_row)
        arguments: List[Argument] = role_row.answer_range
        yield Role(question, tuple(arguments))


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("sys_path")
    ap.add_argument("ground_truth_path")
    ap.add_argument("-s","--sentences_path", required=False)
    args = ap.parse_args()
    main(args.sys_path, args.ground_truth_path, args.sentences_path)
