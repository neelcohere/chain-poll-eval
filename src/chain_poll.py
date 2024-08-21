# ../src/chain_poll.py

import os
import re
import pandas as pd

from pprint import pprint
from concurrent.futures import ThreadPoolExecutor
from ratelimit import limits, sleep_and_retry
from datetime import datetime
from typing import List, Optional, Literal
from tqdm import tqdm
from ast import literal_eval
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from ragas.metrics import faithfulness
from ragas import evaluate
from datasets import Dataset

from src.base import BaseEval
from src.constants import Constant as c

from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")


# chain poll eval prompt
CP_PROMPT = \
"""
You are an LLM Hallucination Evaluator tasked with judging whether the provided Response is relevant to Document the provided Context to ensure that there is no hallucination and that the response adheres to the context provided.

## Context
{context}

## Response
{response}

## Task
Produce a judgement that compare the relevance of the context to the response. For each judgement:
1. Think step by step and check if the claims made by the Response are fully supported by the documents in the Context. 
2. First analyze each document with detailed reasoning for EACH of the documents in context including how it does or does not support the response. Respond with this in the "reasonings" key in the JSON. Make sure ALL documents are included.
3. Then perform an overall analysis summarizing the results across all documents. Respond with this in the "summary" key in the JSON
4. Based on the results and analysis, also include a Yes (if supported) or No (if not supported) if the reponse is fully supported by looking at all the documents. Include this as the "judgement" key in the JSON. 

Repeat the above steps 5 times (5 records) so that we get diverse reasoning and we can average the results across 5 runs. 
"""


# setup pydantic classes for json output
class Reasoning(BaseModel):
    document: int = Field(description="document number from the provided context")
    reasoning: str = Field(description="detailed reasoning step by step for this document to evaluate if it supports the Response or not")


class Record(BaseModel):
    run: int = Field(description="the run number of this record (start from 1 and increment for each additional record)")
    reasonings: List[Reasoning] = Field(description="summarized reasoning for this judgement by individually evaluating the relevance of the Response against EACH documents in Context. Think step-by-step and provide verbose, detailed reasoning to explain you judged the relevance.")
    summary: str = Field(description="final reasoning explanation that summarizes all reasonings for all documents")
    judgement: str = Field(description="Yes if the reasoning indicates that the Response is relevant to and supported by the Context or else No")


class Records(BaseModel):
    records: List[Record] = Field(description="A list of Record JSONs that captures diverse reasonings and judgements for a given Response")


SourceType = Literal["drop", "hotpot", "msmarco", "narrativeqa"]


class ChainPollEval(BaseEval):
    """
    Chain Poll evaluation class
    """
    TOTAL_RUNS = 5

    def __init__(self, *, api_key: str, model: str):
        super().__init__()
        self.api_key = api_key
        self.model = ChatOpenAI(
            api_key=self.api_key, model=model
        )

    def load_data(self, *, path: str | os.PathLike, samples: Optional[int]=None, subset: Optional[SourceType]=None) -> None:
        if samples is not None and subset is not None:
            self.dataset = pd.read_csv(path)
            self.dataset = self.dataset[self.dataset["source"] == subset.lower()].sample(samples)
        if samples is not None and subset is None:
            self.dataset = pd.read_csv(path).sample(samples)
        if samples is None and subset is not None:
            self.dataset = pd.read_csv(path)
            self.dataset = self.dataset[self.dataset["source"] == subset.lower()]
        if samples is None and subset is None:
            self.dataset = pd.read_csv(path)
        cols_to_keep = [
            "unrendered_prompt",
            "generations",
            "prompted_evaluator_score_command-r_generations",
            "source",
        ]
        drop_columns = [col for col in self.dataset.columns if col not in cols_to_keep]
        self.dataset = self.dataset.drop(columns=drop_columns)
        # apply data cleaning
        self.dataset["record_no"] = range(1, self.dataset.shape[0] + 1)
        self.dataset["question"] = self.dataset["unrendered_prompt"].apply(self._extract_question)
        self.dataset["unrendered_prompt"] = self.dataset["unrendered_prompt"].apply(self._extract_unrendered_prompt)
        self.dataset = self.dataset.dropna(subset=["unrendered_prompt"])
        self.dataset["generations"] = self.dataset["generations"].apply(self._extract_generation)
        self.dataset["judgement_prompt"] = self.dataset.apply(self._create_judgement_prompt, axis=1)
        self.dataset["prompted_evaluator_score_command-r_generations"] = \
            self.dataset["prompted_evaluator_score_command-r_generations"].apply(self._extract_evaluator_score)
        self.dataset = self.dataset.rename(columns={
            "generations": "response",
            "unrendered_prompt": "context",
            "prompted_evaluator_score_command-r_generations": "score",
        })
        self.dataset["doc_type"] = self.dataset["context"].apply(lambda x: "multi" if len(x) > 1 else "single")
        
    def run_eval(self, parallel: bool=False) -> None:
        # generate judgement for each record
        parser = JsonOutputParser(pydantic_object=Records)
        cp_prompt = PromptTemplate(
            template="{prompt}" + "\n## Format instructions\n{format_instructions}",
            input_variables=["prompt"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        self.judgement_chain = cp_prompt | self.model | parser
        self.dataset["record"] = None

        if parallel:
            records = self._generate_parallel_judgement()
        else:
            records = []
            for index in tqdm(list(self.dataset["judgement_prompt"].index), desc='Generating judgements', unit="row"):
                # generate judgement
                _record = self._generate_judgement(
                    prompt=self.dataset.loc[index, "judgement_prompt"]
                )
                records.append(_record)

        # add to dataset
        self.dataset["record"] = records

        # calculate Chain Poll score
        self.dataset["cp_score"] = self.dataset.apply(self._calculate_cp_score, axis=1)

        # apply ragas evaluation
        ragas_args = {
            "question": self.dataset['question'].to_list(),
            "answer": self.dataset['response'].to_list(),
            "contexts": self.dataset['context'].to_list(),
        }
        self.dataset["ragas_score"] = self._calculate_ragas_score(**ragas_args)

    @staticmethod
    def get_report(path: str | os.PathLike) -> pd.DataFrame:
        """
        Loads a report from a saved eval run

        Args:
            path (str | PathLike): path to a file or directory. If directory path provided, generates based on latest
                file. specific file can be provided to overwrite
        """
        cols = ["record_no", "source", "doc_type", "question", "response", "score", "cp_score", "ragas_score"]
        # check if a jsonl file is provided
        if os.path.isfile(path) and path.endswith(".jsonl"):
            return pd.read_json(path, lines=True)[cols]
        # check if a directory path is provided
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory or file {path} does not exist")
        runs = [f for f in os.listdir(path) if f.endswith(".jsonl")]
        if not runs:
            raise FileNotFoundError("No runs exist. Run and save evaluation before running the report")
        run_paths = [os.path.join(path, f) for f in runs]
        latest_run_path = max(run_paths, key=os.path.getmtime)
        return pd.read_json(latest_run_path, lines=True)[cols]

    def save_records(self, *, save_dir: str | os.PathLike) -> None:
        if self.dataset is None:
            raise ValueError("Load dataset first")
        if not os.path.isdir(save_dir):
            raise FileNotFoundError(f"Directory {save_dir} does not exist")
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        path = os.path.join(save_dir, "run_" + timestamp + ".jsonl")
        self.dataset.to_json(path, lines=True, orient="records")

    def get_record_info(self, record_no: int) -> list:
        if self.dataset is None:
            raise ValueError("Load dataset first")
        row = self.dataset.loc[self.dataset["record_no"] == record_no]
        question = row["question"].values
        response = row["response"].values
        records = row["record"].values["records"]
        source = row["source"].values
        print(f"QUESTION - {question}")
        print(f"RESPONSE - {response}")
        print(f"SOURCE - {source}")
        for record in records:
            pprint(record)
        return records

    def _calculate_cp_score(self, x: pd.Series) -> float:
        yes_records = len([item for item in x["record"]["records"] if item["judgement"].lower() == "yes"])
        # calculate chain poll score
        cp_score = yes_records / self.TOTAL_RUNS
        return cp_score
    
    def _calculate_ragas_score(self, question: list, answer: list, contexts: list) -> pd.Series:
        data_samples = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
        }
        dataset = Dataset.from_dict(data_samples)
        scores = evaluate(dataset, llm=self.model, metrics=[faithfulness]).to_pandas()
        return scores["faithfulness"].to_list()

    def _generate_judgement(self, prompt: str) -> dict:
        judgement_params = {
            "prompt": prompt
        }
        judgement_record = self.judgement_chain.invoke(
            judgement_params
        )
        return judgement_record
    
    def _generate_parallel_judgement(self) -> list:
        records = []
        # define rate limit
        @sleep_and_retry
        @limits(calls=5, period=100)
        def rate_limited_invoke_call(prompt: str) -> dict:
            
            return self.judgement_chain.invoke(
                {
                    "prompt": prompt
                }
            )

        MAX_THREADS = 5
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            # create list of futures
            futures = [
                executor.submit(
                    rate_limited_invoke_call,
                    prompt=prompt,
                )
                for prompt in self.dataset.loc[:, "judgement_prompt"]
            ]

            # run futures w progress bar
            with tqdm(total=len(futures), desc="Generating judgements") as pbar:
                for future in futures:
                    _record = future.result()
                    records.append(_record)
                    pbar.update(1)
        return records

    def _create_judgement_prompt(self, x: pd.Series) -> str:
        context = ""
        if len(x["unrendered_prompt"]) == 1:
            context = x["unrendered_prompt"][0]
        else:
            for doc in x["unrendered_prompt"]:
                context += f"- {doc}\n"

        prompt = CP_PROMPT.format(
            context = context,
            response = x["generations"]
        )
        return prompt
    
    def _extract_question(self, x: str) -> str:
        # Using re.search to find the question directly
        match = re.search(r'Question:\s*(.*?)(?=\n|$)', x, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    def _extract_unrendered_prompt(self, x: str) -> str:
        # extract the context text
        context_match = re.search(r'Context:\s*(.*?)\s*Task:', x, re.DOTALL)
        if context_match:
            context_text = context_match.group(1).strip()
            # check if the context contains dashed bullet points
            if re.search(r'-\s', context_text):
                # extract dashed bullet points
                bullet_points = re.findall(r'-\s(.*?)(?=\n-|\nTask:|$)', context_text, re.DOTALL)
                context_text = [point.strip() for point in bullet_points]
                return context_text
            else:
                return [context_text]

    def _extract_evaluator_score(self, x: str) -> bool:
        # convert string to Python object
        list_of_lists = literal_eval(x)
        # access the inner list and the boolean string
        boolean_str = list_of_lists[0][0]
        return boolean_str == "True"
    
    def _extract_generation(self, x: str) -> str:
        # convert string to Python object
        generation_list = literal_eval(x)
        # access the generation
        generation = generation_list[0]
        return generation
