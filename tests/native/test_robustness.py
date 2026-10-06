from llamatestclient import llm_client


def test_input_variation(llm_client):
    prompts = ["Whats the capital of Italy", "What is the capital of Italy",
               "What is the capital of italy", "Italy capital?]"]

    for prompt in prompts:
        response = llm_client.generate(prompt)

        assert "rome" in response["text"].lower(), "Output is incorrect as Rome isnt in answer"

