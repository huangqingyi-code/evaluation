import os
import re
import ast
from typing import List, Dict, Any
from recall_eval.prompt import prompt_sys, prompt_user
from recall_eval.eval_metrics import Metric
from utils import save_json


def pprint_format(result):
    print("tables:")
    print("macro-Averaged Recall", result["table"]["macro-Averaged Recall"])
    print("macro-Averaged F1", result["table"]["macro-Averaged F1"])
    print("columns:")
    print("macro-Averaged Recall", result["column"]["macro-Averaged Recall"])
    print("macro-Averaged F1", result["column"]["macro-Averaged F1"])


def make_pred(samples, pred_ext):
    assert len(samples) == len(pred_ext)
    preds = []
    for i in range(len(samples)):
        preds.append(
            {
                "query": samples[i]["query"],
                "output": str(pred_ext[i]["output"]),
                "pred_table": str(pred_ext[i]["tables"]),
                "pred_col": str(pred_ext[i]["columns"]),
                "label_table": samples[i]["label_table"],
                "label_col": samples[i]["label_col"],
            }
        )
    return preds


def format_inputs(samples: List[Dict]) -> List:
    """
    输入数据格式化函数，按照 generate 的格式要求改造 inputs
    :param samples: 待格式化样例数据
    :param mode: 格式化模式
    """
    # 把需要推理的数据拼成 message 形式
    msgs = []
    for sample in samples:
        msg_sys = prompt_sys
        msg_user = prompt_user.format(
            table_infos=sample["table_infos"], query=sample["query"]
        )
        msg = [
            {"role": "system", "content": msg_sys},
            {"role": "user", "content": msg_user},
        ]
        msgs.append(msg)
    return msgs


def parser_text(text: str) -> Any:
    """
    llm 推理结果解析函数，提取 生成代码 或 召回的表格和字段信息
    :param text: 文本，形如 llm_response['output_text']
    :param mode: 解析模式，gen 为解析生成的代码，extract 为解析召回结果
    """
    pattern_table = r"(?i)tables(?:\s+is)?\s*:\s*\[([^\]]+)\]"
    pattern_column = r"(?i)columns(?:\s+is)?\s*:\s*\[([^\]]+)\]"
    text = text.replace("【", "[").replace("】", "]").replace("`", "")
    match_tables = re.findall(pattern_table, text.strip())
    match_columns = re.findall(pattern_column, text.strip())
    tables = []
    columns = []
    if match_tables:
        try:
            tables = ast.literal_eval(f"[{match_tables[0]}]")
        except Exception as e:
            tables = []
    if match_columns:
        try:
            columns = ast.literal_eval(f"[{match_columns[0]}]")
            if len(tables) == 1 and len(columns) > 0:
                columns = [
                    (
                        "{}.{}".format(tables[0], column)
                        if not column.startswith("{}.".format(tables[0]))
                        else column
                    )
                    for column in columns
                ]
        except Exception as e:
            columns = []
    return {"tables": tables, "columns": columns, "output": text}


def parser_list(batch_resp):
    # 批处理解析
    return [parser_text(resp["output_text"]) for resp in batch_resp]


def save_result(preds, report, test_file_path):
    # 保存 LLM 生成的内容 和 最终的 recall_eval 结果
    parent_path = os.path.dirname(test_file_path)
    save_path = os.path.join(parent_path, "recall_eval_llm_gen_data.json")
    save_json(save_path, preds)
    print(f"Recall Eval Saved:{save_path}")
    save_path = os.path.join(parent_path, "recall_eval_report.json")
    save_json(save_path, report)
    print(f"Recall Eval Saved:{save_path}")


def eval_outputs(preds: List[Dict], samples: List[Dict], lang: str = None) -> Dict:
    """
    eval结果计算函数，使用 Metric 中评估方法，评估表格、字段召回的相关指标
    :param preds: 模型预测结果
    :param samples: 数据集测试样本
    """

    def combine_metrics_under_key(pred_data, label_data, key):
        combined_metrics = {}
        # for metric_name in ["averaged", "jaccard", "hamming"]:
        for metric_name in ["averaged"]:
            metric_results = getattr(Metric, metric_name)(pred_data, label_data)
            combined_metrics.update(metric_results)

        return {key: combined_metrics}

    pred_tables = [pred["tables"] for pred in preds]
    pred_columns = [pred["columns"] for pred in preds]
    label_tables = [sample["label_table"] for sample in samples]
    label_columns = [sample["label_col"] for sample in samples]
    if lang == "python":
        label_columns = [[col.split(".")[1] for col in cols] for cols in label_columns]
    table_metrics_combined = combine_metrics_under_key(
        pred_tables, label_tables, "table"
    )
    column_metrics_combined = combine_metrics_under_key(
        pred_columns, label_columns, "column"
    )
    merged_results = {**table_metrics_combined, **column_metrics_combined}
    return merged_results
