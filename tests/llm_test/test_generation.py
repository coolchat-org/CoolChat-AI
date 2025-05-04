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
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

dataset = EvaluationDataset(test_cases=make_testcases("try.xlsx"))


@pytest.mark.parametrize("test_case", dataset.test_cases)
def test_generation(test_case: LLMTestCase):
    """
    Với mỗi test_case, assert_test sẽ:
      - so sánh actual_output vs. expected_output
      - tính score trên mỗi metric
      - raise AssertionError nếu bất kỳ metric nào fail
    """ 
    metrics = [
        AnswerRelevancyMetric(threshold=0.7, include_reason=False),
        FaithfulnessMetric(threshold=0.7, include_reason=False)
    ]
    assert_test(test_case, metrics)

import deepeval

@deepeval.on_test_run_end
def after_test_run():
    print("✅ Finished Generation testing")
