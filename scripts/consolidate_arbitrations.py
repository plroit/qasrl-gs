from argparse import ArgumentParser

import pandas as pd
from sklearn.utils import shuffle
import os


def main(arbit_path: str):
    df = pd.read_csv(arbit_path)
    ids = df[['qasrl_id', 'verb_idx', 'assign_id']].drop_duplicates()
    ids = shuffle(ids)
    single_assign_ids = ids.groupby(['qasrl_id', 'verb_idx']).head(1)
    selected_df = pd.merge(df, single_assign_ids, on=['qasrl_id', 'verb_idx', 'assign_id'])
    is_accepted = selected_df.answer_range.notnull()
    selected_df = selected_df[is_accepted].copy()
    out_path = get_out_path(arbit_path)
    selected_df.to_csv(out_path, index=False, encoding="utf-8")


def get_out_path(arbit_path: str):
    dir_name, file_name = os.path.split(arbit_path)
    base_file_name = os.path.splitext(os.path.splitext(file_name)[0])[0]
    out_file_name = f"{base_file_name}.silver.csv"
    out_path = os.path.join(dir_name, out_file_name)
    return out_path

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("arbit_path")
    args = ap.parse_args()
    main(args.arbit_path)
