# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/25 15:39
@Auth ： zhaliangyu
@File ：run_eval.py
@IDE ：PyCharm
"""
import signal
import pandas as pd
import json
import os
import datetime
from tqdm import tqdm
from typing import Optional, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from evaluate_code_correction.utils import filter_code, filter_cot, get_tool
from evaluate_code_correction.prompt import (
    RECTIFY_PROMPT_PYTHON_SYSTEM,
    RECTIFY_PROMPT_PYTHON_INSTRUCTION,
    CLASSIFY_PROMPT_PYTHON,
)

from contextlib import contextmanager

# 定义一个异常类，用于超时处理
class TimeoutException(Exception):
    pass

# 创建一个上下文管理器来处理超时
@contextmanager
def timeout(time):
    # 定义信号处理函数
    def raise_timeout(signum, frame):
        raise TimeoutException(f"Timeout error, running time exceed {time}")

    # 设置信号定时器
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(time)
    try:
        yield
    finally:
        # 取消信号定时器
        signal.alarm(0)

def llm_eval(
    query: str,
    code: str,
    observation: str,
    true_result,
    llm: BaseLanguageModel,
) -> bool:
    """
    Compare the eval_llm output results and true_results by a Higher-level LLM, `gpt-4` etc, make this llm config in llms.py
    :param query: Human input query
    :param table_infos: Input table information
    :param code: Code generated by the llm_eval
    :param observation: Code execution result
    :param llm: llm_judge
    :return: Enum [True, False]
    """
    print("True result", true_result)
    prompt = ChatPromptTemplate.from_messages(
        [("system", CLASSIFY_PROMPT_PYTHON)]
    )
    eval_chain = prompt | llm | StrOutputParser()
    # eval_chain.verbose = True
    input = {
        "query": query,
        # "table_infos": table_infos,
        "code": code,
        "observation": observation,
        "true_result": true_result
    }
    output = eval_chain.invoke(
        input=input
    )
    res = output
    print("Observe:", observation)
    print("LLM eval results: ", res)
    return True if res.lower() == "yes" else False


def format_inputs(
    test_datas: list[dict],
    lan_type: str = "Python"
) -> list[list[dict]]:
    """
    Format inputs with prompts and input variances
    :param test_datas: loaded eval samples
    :param lan_type: Code type, support [`Python`] now
    :return
    """
    # 把需要推理的数据拼成 message 形式
    format_message_datas = []
    for idx, sample in tqdm(enumerate(test_datas)):
        queries = sample["query"]
        table_infos = sample["table_infos"]

        current_time = datetime.datetime.now().strftime('%Y-%m-%d:%H')
        output = sample["cot"] + f"{lan_type} Code:\n" + sample["code"]
        observes = sample["observation"]

        format_instruction = RECTIFY_PROMPT_PYTHON_INSTRUCTION.format(
            table_infos=table_infos,
            query=queries,
            observe=observes,
            current_time=current_time,
            output=output
        )
        format_system = RECTIFY_PROMPT_PYTHON_SYSTEM
        messages = [
            {"role": "system", "content": format_system},
            {"role": "user", "content": format_instruction}
        ]
        format_message_datas.append(messages)

    return format_message_datas

def eval_outputs(
    model_outputs: list[dict],
    eval_dataset_path: str,
    test_csv_file_path: str,
    lan_type: str
) -> list[dict]:
    """
    Generate complete eval samples according to the eval_datasets
    and the model_outputs
    :param model_outputs: output_answers generate by the llm
    :param eval_dataset_path: eval dataset path
    :param test_csv_file_path: the csv files path
    :param
    :return Required complete output_answers List[Dict]
    """
    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        test_datas = json.load(f)
    output_texts = [i["output_text"] for i in model_outputs]
    processed_data = []
    for idx, test_dt in enumerate(test_datas):
        llm_output = output_texts[idx]
        table_infos = test_datas[idx]["table_infos"]
        df_paths = test_datas[idx]["table_paths"]
        true_result = test_datas[idx]["true_result"]
        query = test_datas[idx]["query"]
        eval_result_sample = {}

        if len(df_paths) == 1:
            df = pd.read_csv(os.path.join(test_csv_file_path, df_paths[0]), low_memory=False)
        else:
            df = [pd.read_csv(os.path.join(test_csv_file_path, path), low_memory=False) for path in df_paths]
        tool = get_tool(df)

        code = filter_code(llm_output)
        cot = filter_cot(llm_output)
        output = cot + f"{lan_type} Code:\n" + code
        try:
            try:
                with timeout(15):  # 设置超时时间为15秒
                    observe = tool.run(code)  # 需要监控超时的代码块
            except TimeoutException as e:
                observe = e
        except SystemExit as e:
            observe = e 
            # 处理 SystemExit 异常，例如记录日志、清理资源等
        except Exception as e:
            observe = e
        eval_result_sample["code"] = output
        eval_result_sample["observe"] = observe
        eval_result_sample["true_result"] = true_result
        eval_result_sample["table_infos"] = table_infos
        eval_result_sample["table_paths"] = df_paths
        eval_result_sample["query"] = query
        processed_data.append(eval_result_sample)
    return processed_data

def execution_eval(output_code: str, dfs: Any) -> bool:
    """
    Test whether the code generated by eval_llm can be executed.
    :param output: output code of llm generation
    :return: True or False
    """
    import re
    python_executor = get_tool(dfs)
    code = output_code.strip(" ").strip('"')
    code_res = python_executor.run(code)
    # 只要执行结果中不出现error 或者 exception， 就认为代码可执行
    pattern = re.compile(r"error|exception", re.IGNORECASE)

    try:
        res = not pattern.search(code_res)
    except:
        res = True
    print("Execute Observe:", code_res)
    print("Execute Result:", res)
    return res

def run_eval(
    eval_result_path: str = "../evalset/code_correction_test/results.json",
    test_csv_file_path: str = "./",
    llm_for_judge: Optional[BaseLanguageModel] = None
):
    """
    Calculate eval pass rate, support execute_pass_rate and llm_eval_pass_rate
    :param eval_results_path:  Evaluation dataset path
    :param llm_for_judge: llm for classify the content generated by the llm_eval, this param is used while `eval_method == "execution",`
    :return: pass rate
    """
        # print(eval_answer)
    import json
    with open(eval_result_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    execute_passed, llm_eval_passed = 0, 0
    total_len = len(samples)
    for sample in tqdm(samples):
        code = filter_code(sample["code"])
        observe = sample["observe"]
        true_result = sample["true_result"]
        df_paths = sample["table_paths"]
        query = sample["query"]
        if len(df_paths) == 1:
            df = pd.read_csv(os.path.join(test_csv_file_path, df_paths[0]), low_memory=False)
        else:
            df = [pd.read_csv(os.path.join(test_csv_file_path, path), low_memory=False) for path in df_paths]
        execute_passed += 1 if execution_eval(code, df) else 0
        if llm_for_judge is not None:
            llm_eval_passed += 1 if llm_eval(query,
                                    code, observe,
                                    true_result, llm_for_judge) else 0
        print("*" * 20)
    print(f"Sample length: {total_len}. "
          f"Execute Passed: {execute_passed}."
          f"Execute pass-rate is:", round(execute_passed / total_len, 3))
    if llm_for_judge is not None:
        print(f"LLM eval Passed: {llm_eval_passed}")
        print(f"LLM_eval pass-rate is:", round(llm_eval_passed / total_len, 3))
