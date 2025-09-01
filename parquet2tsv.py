#!/usr/bin/env python3
import sys, base64, ast, pandas as pd

def bbox2str(bbox):
    if isinstance(bbox, str):
        bbox = ast.literal_eval(bbox)
    return str(list(map(int, bbox)))

def main(parquet_path, tsv_path):
    df = pd.read_parquet(parquet_path)
    rows = []
    for _, r in df.head(20).iterrows():  # 只取前 20 行
        idx   = int(r.question_id)
        q     = str(r.question).strip()
        img   = base64.b64encode(r.image['bytes']).decode()
        ans   = bbox2str(r.bbox)
        rows.append([idx, img, q, ans])

    pd.DataFrame(rows, columns=['index', 'image', 'question', 'answer']) \
      .to_csv(tsv_path, sep='\t', index=False, encoding='utf-8')
    print(f'Done! TSV (20 rows) saved to {tsv_path}')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python parquet2tsv.py input.parquet output.tsv')
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])