import time
from urllib import response

import pytest
import requests


class OllamaTestClient:
    def __init__(self, model="tinyllama", host="localhost", port=11434):
        self.model = model
        # self.host = host
        self.endpoint = f"http://{host}:{port}/api/generate"

    def check_model_availability(self):
        try:
            response = requests.post(self.endpoint, json={
                "model": self.model,
                "prompt": "test",
                "stream": False
            })
            return response.status_code == 200
        except:
            return False

    def generate(self, prompt, temperature=0.8, max_tokens=1000):
        start_time = time.time()
        response = requests.post(self.endpoint, json={
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        })
        end_time = time.time()
        if response.status_code != 200:
            raise Exception("API call failed:" + response.text)
        result =  response.json()
        return {
            "text": result["response"],
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(result["response"].split()),
            "latency": end_time - start_time
        }

@pytest.fixture
def llm_client():
    client = OllamaTestClient(model="tinyllama")
    if not client.check_model_availability():
        pytest.skip("Tinyllama model not available, please run 'Ollama pull tinyllama' to avail it")

    return client