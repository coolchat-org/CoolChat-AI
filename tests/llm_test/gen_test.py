import pandas as pd
from deepeval.test_case import LLMTestCase

def make_testcases(filename: str = "llmtest.xlsx"):
    df = pd.read_excel(filename)
    separator = '-' * 40

    test_cases = []
    for _, row in df.iterrows():
        tc = LLMTestCase(
            input=row["Input"],
            expected_output=row["Expected Output"],
            actual_output=row["Actual output"],
            retrieval_context=[part.strip() for part in row.get("Context", None).split(separator)] 
        )
        test_cases.append(tc)

    return test_cases