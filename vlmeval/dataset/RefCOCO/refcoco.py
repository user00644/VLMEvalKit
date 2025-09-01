import json, os, re
from ..image_base import ImageBaseDataset
from ...smp import *      # 官方 util

class RefCOCO(ImageBaseDataset):
    TYPE = 'Visual_Grounding'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print('[RefCOCO] 我被加载啦！')

    @classmethod
    def supported_datasets(cls):
        return ['RefCOCO']      # 必须返回包含 'RefCOCO' 的列表


    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value= f"Locate {question}, output its bbox coordinates using JSON format."))
        return msgs


    def evaluate(self, eval_file, **judge_kwargs):
        df = load(eval_file)

        # 2. 解析预测：支持 JSON 列表或 <box> 标签
        # print(df['answer'])
        gts = df['answer'].apply(self._load_gt).tolist()
        # print("answers:", answers)

        preds = df['prediction'].apply(self._parse_box).tolist()
        # print("preds:", preds)

        # 3. 计算指标
        ious = [self._iou(g, p) for g, p in zip(gts, preds)]
        acc  = sum(i >= 0.5 for i in ious) / len(ious) if ious else 0
        return {'Acc@0.5': acc, 'mIoU': sum(ious) / len(ious) if ious else 0}


    def _load_gt(self, data: str):
        """把任意字符串 -> [x1,y1,x2,y2]，永不抛异常"""
        import ast
        data = str(data).strip()
        if not data or data.lower() in {'nan', 'none', ''}:
            return [0, 0, 0, 0]
        try:
            box = ast.literal_eval(data)
            if isinstance(box, list) and len(box) == 4:
                return list(map(int, box))
        except Exception:
            pass

    def _parse_box(self, json_str: str):
        try:
            data = re.sub(r'^```json\s*|\s*```$', '', json_str, flags=re.S).strip()
            data = json.loads(data)
            return data[0]['bbox_2d']
        except Exception:
            return [0, 0, 0, 0]


    @staticmethod
    def _iou(a, b):
        """计算 IoU，a、b 均为 [x1,y1,x2,y2]"""
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / union if union else 0
