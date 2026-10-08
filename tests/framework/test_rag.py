from deepeval import assert_test
from deepeval.evaluate import evaluate
from deepeval.metrics import HallucinationMetric
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from openaitestclient import OpenAITestClient

openai_client = OpenAITestClient(model="gpt-4.1-nano")
evalution_model = OpenAIModel(model="gpt-4o-mini")

actual_output = "We have a 30 day return policy will full refund"

retrieval_context = ["All customers are eligible for full refund until 30 days with no extra cost"]

contextual_relevancy_metric =ContextualRelevancyMetric(
    model=evalution_model,
    threshold=0.7,
    include_reason=True
)

test_case = LLMTestCase(
    input="What if i dont like the shoes i buy?",
    actual_output=actual_output,
    retrieval_context=retrieval_context
    )

evaluate(test_cases=[test_case], metrics=[contextual_relevancy_metric])
