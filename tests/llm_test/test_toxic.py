import dotenv
from gen_test import make_testcases
dotenv.load_dotenv()
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import BiasMetric, ToxicityMetric


dataset = EvaluationDataset(test_cases=make_testcases())


@pytest.mark.parametrize("test_case", dataset.test_cases)
def test_toxicity(test_case: LLMTestCase):
    """
    Với mỗi test_case, assert_test sẽ:
      - so sánh actual_output vs. expected_output
      - tính score trên mỗi metric
      - raise AssertionError nếu bất kỳ metric nào fail
    """
    metrics = [ToxicityMetric()]
    assert_test(test_case, metrics)

import deepeval

@deepeval.on_test_run_end
def after_test_run():
    print("✅ Finished Toxicity testing")
