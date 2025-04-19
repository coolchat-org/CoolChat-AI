from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.dataset import EvaluationDataset
import pytest
# import os, dotenv
from deepeval.models import GPTModel

# dotenv.load_dotenv()



# first_test = LLMTestCase(input="Các đặc điểm của web3 là gì?")
# dataset = Evaluation
# @pytest.mark.parametrize(
#     "test_case",
#     dataset
# )

answer_relevancy = AnswerRelevancyMetric()
faithfulness = FaithfulnessMetric()
test_case = LLMTestCase(
    input="I'm on an F-1 visa, gow long can I stay in the US after graduation?",
    actual_output="You can stay up to 30 days after completing your degree.",
    expected_output="You can stay up to 60 days after completing your degree.",
    retrieval_context=[
        """If you are in the U.S. on an F-1 visa, you are allowed to stay for 60 days after completing
        your degree, unless you have applied for and been approved to participate in OPT."""
    ]
)
answer_relevancy.measure(test_case)
print("Score: ", answer_relevancy.score)
print("Reason: ", answer_relevancy.reason)

faithfulness.measure(test_case)
print("Score: ", faithfulness.score)
print("Reason: ", faithfulness.reason)