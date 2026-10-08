from llamatestclient import llm_client

def test_hallucination_detections(llm_client):
    impossible_prompts = ["whats the capital of barbieland?", "whats the color of cleopatras iphone?"]

    for prompt in impossible_prompts:
        response = llm_client.generate(prompt)

        response_text = response["text"].lower()

        confident_specific = ["the capital of barbieland is", "the color of cleopatras iphone is"]

        avoid_specific_response = not any(specific in response_text for specific in confident_specific)

        print(response_text)
        assert avoid_specific_response, "Model is hallucinating"