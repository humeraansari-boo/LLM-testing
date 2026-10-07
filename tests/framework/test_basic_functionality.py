from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from openaitestclient import OpenAITestClient

openai_client = OpenAITestClient(model="gpt-4.1-nano")
evalution_model = OpenAIModel(model="gpt-4o-mini")

def test_basic_response():

    prompt = "Hello, how are you?"

    response = openai_client.generate(prompt)
    test_case = LLMTestCase(input=prompt,actual_output=response["text"])

    response_quality_metric = GEval(name="Evaluate response quality",
          criteria="The response should be sensical, not empty, have more than 10 characters and be appropriate greeting response",
          evaluation_params = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
          threshold=0.7,
          model=evalution_model
          )

    assert_test(test_case, [response_quality_metric])

def test_instruction_following():
    prompt = "Name 3 colors, only list them"
    response = openai_client.generate(prompt)
    test_case = LLMTestCase(input=prompt,actual_output=response["text"])

    response_quality_metric = GEval(name="Evaluate instruction quality",
                                    criteria="Check if the response correctly names 3 different colors. The response should be sensical, not empty, have more than 10 characters and be appropriate greeting response",
                                    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
                                    threshold=0.7,
                                    model=evalution_model
                                    )

    assert_test(test_case, [response_quality_metric])

def test_simple_qa():
    prompt = "Whats the capital of Italy?"
    response = openai_client.generate(prompt)

    response_quality_metric = GEval(
        name="Evaluate correct response",
        criteria="Check if the response correctly names Rome as the capital of Italy",
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        threshold=0.8,
        model=evalution_model
        )
    test_case = LLMTestCase(input=prompt,
                            actual_output=response["text"],
                            expected_output="The capital of Italy is Rome")
    assert_test(test_case, [response_quality_metric])