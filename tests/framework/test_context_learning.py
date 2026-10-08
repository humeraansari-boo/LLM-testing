from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from openaitestclient import OpenAITestClient

openai_client = OpenAITestClient(model="gpt-4.1-nano")
evalution_model = OpenAIModel(model="gpt-4o-mini")


def test_logical_consistency():
    equivalent_questions = [("Whats is the capital of Italy?", "Rome is the capital of which country?")]

    for q1,q2 in equivalent_questions:
        response1 = openai_client.generate(q1)
        response2 = openai_client.generate(q2)

        response_quality_metric = GEval(name="Evaluate logical consistency",
                                        criteria="Evaluate the logical consistency of both the responses:"
                                                 "1. make sure they are both factually correct and consistent"
                                                 "2. both should address same underlying concept"
                                                 "3. no condradictory information should be present"
                                                 "4. differently phrased responses is okay"
                                                 "Award high scores to the responses only when both the responses "
                                                 "are factually aligned  ",
                                        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
                                        threshold=0.7,
                                        model=evalution_model
                                        )
        response_text1 = response1["text"]
        response_text2 = response2["text"]
        combined_input = "Question 1: " + q1 + "\n" + "Question 2: " + q2 + "\n"
        combined_output = "Response 1: " + response_text1 + "\n" + "Response 2: " + response_text2 + "\n"

        test_case = LLMTestCase(input=combined_input, actual_output=combined_output)
        assert_test(test_case, [response_quality_metric])
