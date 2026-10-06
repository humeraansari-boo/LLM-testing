from llamatestclient import llm_client
import re


def test_list_format(llm_client):
    response = llm_client.generate("List only 3 fruits, format answers in a numbered list")

    list_pattern = r'[1-3][.)]'
    matches = re.findall(list_pattern, response['text'])
    print("Response: ", response['text'])
    assert len(matches) >= 1, "Response doesn't contain a numbered list"
    print(f"Found: {len(matches)} list markers in response")

def test_basic_json(llm_client):
    response = llm_client.generate("Create a json with name Loki and age 7")
    assert "{" in  response['text'] and "}" in response['text'], "Response contains the json braces"
    assert "name" in response['text'] and "loki" in response['text'].lower(), "Response contains the name"
    assert "age" in response['text'] and "7" in response['text'], "Response contains the age"
    print("Response: ", response['text'])
