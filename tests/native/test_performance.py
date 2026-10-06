import time

from llamatestclient import llm_client

def test_basic_functionality(llm_client):
    prompt = "Hello"

    start_time = time.time()

    responses = []

    for _ in range(3):
        response = llm_client.generate(prompt, max_tokens=20)
        responses.append(response)

    end_time = time.time()

    total_time = end_time - start_time
    total_tokens = sum(r["completion_tokens"] for r in responses)
    tokens_per_sec = total_tokens / total_time
    requests_per_min = (3/total_time) * 60

    print("Total time: ", total_time)
    print("Total tokens: ", total_tokens)
    print("Requests per minute: ", requests_per_min)
    print("Percentage completion: ", tokens_per_sec)