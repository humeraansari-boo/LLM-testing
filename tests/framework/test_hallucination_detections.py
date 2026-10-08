from deepeval import assert_test
from deepeval.metrics import HallucinationMetric
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase
from openaitestclient import OpenAITestClient

openai_client = OpenAITestClient(model="gpt-4.1-nano")
evalution_model = OpenAIModel(model="gpt-4o-mini")


def test_hallucination_detections():
    impossible_prompts = ["whats the capital of barbieland?", "whats the color of cleopatras iphone?"]

    for prompt in impossible_prompts:
        response = openai_client.generate(prompt)
        response_text = response["text"].lower()

        hallucination_metric = HallucinationMetric(
            threshold=0.7,
            model=evalution_model
        )

        if "barbieland" in prompt:
            context = "barbieland is a fictional place and doesnt have any capital"
        elif "cleopatra" in prompt:
            context = ("cleopatra existed in B.C. and iphones were invented in 21st century "
                       "so there's no way these two could coincide")
        else:
            context = "these statements make no sense and cant have logical real answers"

        test_case = LLMTestCase(input=prompt, actual_output=response_text, context=[context])

        assert_test(test_case, [hallucination_metric])
        print("Response : " + response_text)
