import os
from argparse import ArgumentParser
import json
import pandas as pd
from decode_encode_answers import encode_qasrl

def load_records(path):
    with open(path, "r", encoding="utf-8") as fin:
        records = [json.loads(line) for line in fin]
        return records


def yield_roles_from_parser(records, min_score):
    for rec in records:
        qasrl_id = rec['qasrl_id']
        for verb in rec['verbs']:
            predicate = verb['verb']
            predicate_idx = verb['index']
            for qa_pair in verb['qa_pairs']:
                question = qa_pair['question']
                slots = qa_pair['slots']
                answer_ranges = [(span['start'], span['end']+1)
                                 for span in qa_pair['spans']
                                 if span['score'] > min_score]
                answers = [span['text'] for span in qa_pair['spans']
                           if span['score'] > min_score]
                if not answer_ranges:
                    continue
                item = {
                    'qasrl_id': qasrl_id,
                    'verb_index': predicate_idx,
                    'verb': predicate,
                    'question': question,
                    'answer': answers,
                    'answer_range': answer_ranges,
                }
                item.update(slots)
                yield item


def main(args):
    records = load_records(args.parser_path)
    records = list(yield_roles_from_parser(records, args.min_score))
    df = pd.DataFrame(records)
    slot_headers = ['wh', 'aux', 'subj', 'obj', 'verb_slot_inflection',
                    'prep', 'obj2', 'is_passive', 'is_negated']
    cols = ['qasrl_id', 'verb_idx', 'verb', 'question', 'answer', 'answer_range', 'wh'] + slot_headers
    out_path = os.path.splitext(args.parser_path)[0] + f".T_{args.min_score}.csv"
    df = df[cols].copy()
    df = encode_qasrl(df)
    df[cols].to_csv(out_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("parser_path")
    ap.add_argument("--min_score", default=0.0, type=float)
    main(ap.parse_args())

