from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from openaitestclient import OpenAITestClient

openai_client = OpenAITestClient(model="gpt-4.1-nano")
evalution_model = OpenAIModel(model="gpt-4o-mini")

def test_list_format():
    prompt = "List only 3 fruits, format answers in a numbered list"

    response = openai_client.generate(prompt)

    response_quality_metric = GEval(name="Evaluate formatting skills of response",
                                    criteria="The response should be sensical, not empty, have more than 10 characters"
                                             "it should also follow: "
                                             "1. Exactly 3 items (fruits)"
                                             "2. Use numbered list formatting"
                                             "3. list should have either {1.} format or {1)} format",
                                    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
                                    threshold=0.8,
                                    model=evalution_model
                                    )
    test_case = LLMTestCase(input=prompt, actual_output=response["text"])

    assert_test(test_case, [response_quality_metric])

    print("Response: ", response['text'])

def test_basic_json():
    prompt = "Create a json with name Loki and age 7"
    response = openai_client.generate(prompt)

    response_quality_metric = GEval(name="Evaluate Basic Json creation skills of response",
                                    criteria="The response should be sensical, not empty, have more than 10 characters"
                                             "it should also follow:"
                                             "1. { and } should be in the response"
                                             "2. name, loki, age and 7 should all be present",
                                    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
                                    threshold=0.8,
                                    model=evalution_model
                                    )
    test_case = LLMTestCase(input=prompt, actual_output=response["text"])
    assert_test(test_case, [response_quality_metric])
    print("Response: ", response['text'])
