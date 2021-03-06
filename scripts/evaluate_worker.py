from argparse import ArgumentParser
import pandas as pd
from decode_encode_answers import decode_qasrl
from evaluate_dataset import eval_datasets
import os

def parse_args():
    ap = ArgumentParser()
    ap.add_argument("qasrl_path", help="/path/to/qasrl_annotation_output.csv")
    ap.add_argument("ref_path", help="/path/to/qasrl_ground_truth.csv")
    ap.add_argument("sent_path", help="/path/to/sentences.csv")
    ap.add_argument("out_dir", help="/path/to/directory_where_a_report_for_each_worker_is_saved")
    return ap.parse_args()


def main():
    args = parse_args()
    qasrl_path = args.qasrl_path
    out_dir = args.out_dir

    qasrl = decode_qasrl(pd.read_csv(qasrl_path))
    ref = decode_qasrl(pd.read_csv(args.ref_path))
    sents = pd.read_csv(args.sent_path)
    sent_map = dict(zip(sents.qasrl_id, sents.tokens.apply(str.split)))

    worker_data = []
    # Step 1: get a dataset for each worker
    work_groups = qasrl.groupby("worker_id")
    for worker_id, worker_df in work_groups:
        print(f"Evaluating: {worker_id}")
        # Step 2: for each worker, compare the dataset
        # with the reference on common predicate ids.
        w_pred_ids = worker_df[['qasrl_id', 'verb_idx']].drop_duplicates()
        w_ref = pd.merge(ref, w_pred_ids, on=['qasrl_id', 'verb_idx'])
        res = eval_datasets(w_ref, worker_df)

        # Step 3: for each worker, get argument precision and recall, and avg. number of questions per verb.
        arg_counts, role_counts, all_matchings = res
        tp, fp, fn = arg_counts
        prec = tp / (tp + fp)
        recall = tp / (tp + fn)
        n_questions = all_matchings.sys_role.nunique()
        n_predicates = w_pred_ids.shape[0]
        qs_per_pred = float(n_questions)/n_predicates
        worker_path = os.path.join(out_dir, f"{worker_id}.csv")

        # Step 5: save report.
        all_matchings.to_csv(worker_path, index=False, encoding="utf-8")
        worker_data.append({
            "worker_id": worker_id,
            "prec": prec,
            "recall": recall,
            "n_preds": n_predicates,
            "qs_per_pred": qs_per_pred})
    worker_data = pd.DataFrame(worker_data)[['worker_id', 'n_preds', 'qs_per_pred', 'prec', 'recall']].copy()

    # Step 4: display result
    print(worker_data.sort_values(['n_preds', 'qs_per_pred'], ascending=False))


if __name__ == "__main__":
    main()