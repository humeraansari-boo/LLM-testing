import os

from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

from tests.framework.shoe_store_rag import ShoeStoreRag


def test_shoe_store_rag_with_deepeval():
    pinecone_key = os.getenv("PINE_CONE_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    rag = ShoeStoreRag(pinecone_key, openai_key)
    test_scenarios = [
        "What if my shoe doesnt fit, can i return it?",
        "Do you offer any discounts?",
        "When will I get my shoe?",
        "Do you have Nike and Adidas?"
    ]

    context_relevancy = ContextualRelevancyMetric(
        threshold=0.7,
        model="gpt-4o-mini",
        include_reason=True
    )

    failed_scenarios = []
    for number, scenario in enumerate(test_scenarios, start=1):
        print(f"\n========== Scenario {number}/{len(test_scenarios)} ==========")
        print("Question:", scenario)

        retrieval_context = rag.retrieve_context(scenario)
        print("Retrieved documents:")
        for doc in retrieval_context:
            print("  -", doc)

        actual_output = rag.generate_answer(scenario, retrieval_context)
        print("Answer:", actual_output)

        test_case = LLMTestCase(
            input=scenario,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )

        # Score this scenario on its own so one failure doesn't stop the others
        context_relevancy.measure(test_case)
        passed = context_relevancy.is_successful()
        print(f"Score: {context_relevancy.score} (threshold: {context_relevancy.threshold})")
        print("Reason:", context_relevancy.reason)
        print("Result:", "PASSED" if passed else "FAILED")

        if not passed:
            failed_scenarios.append(scenario)

    print("\n========== Summary ==========")
    print(f"Passed: {len(test_scenarios) - len(failed_scenarios)}/{len(test_scenarios)}")
    for scenario in failed_scenarios:
        print("  FAILED:", scenario)

    assert not failed_scenarios, f"{len(failed_scenarios)} scenario(s) failed: {failed_scenarios}"
