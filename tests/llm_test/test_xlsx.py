import dotenv
from gen_test import make_testcases
dotenv.load_dotenv()
import pandas as pd
import pytest
import os
os.environ["DEEPEVAL_RESULTS_FOLDER"] = "./test_results"
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import ContextualRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric

dataset = EvaluationDataset(test_cases=make_testcases(filename="try.xlsx"))


@pytest.mark.parametrize("test_case", dataset.test_cases)
def test_retrieval(test_case: LLMTestCase):
    """
    Deepeval offers three LLM evaluation metrics to evaluate retrievals:

    - ContextualPrecisionMetric: evaluates whether the reranker in your retriever ranks more relevant nodes in your retrieval context higher than irrelevant ones.

    - ContextualRecallMetric: evaluates whether the embedding model in your retriever is able to accurately capture and retrieve relevant information based on the context of the input.

    - ContextualRelevancyMetric: evaluates whether the text chunk size and top-K of your retriever is able to retrieve information without much irrelevancies.
    """
    metrics = [
        ContextualRelevancyMetric(),
        ContextualPrecisionMetric(),
        ContextualRecallMetric()
    ]
    assert_test(test_case, metrics)

import deepeval

@deepeval.on_test_run_end
def after_test_run():
    print("✅ Finished testing, results are saved in ./test_results")
