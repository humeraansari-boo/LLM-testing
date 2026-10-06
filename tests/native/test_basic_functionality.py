from llamatestclient import llm_client



def test_basic_response(llm_client):
    response = llm_client.generate("Why is sky blue?")

    assert response['text'].strip() != '', "response text should not be empty"
    assert len(response['text']) > 10, "response text should be longer than 10"

    print("Basic response of prompt is of length: ", len(response['text']))


def test_instruction_following(llm_client):
    response = llm_client.generate("Name 3 colors, only list them")
    response_text = response['text'].lower()

    color_bank = ["red", "green", "blue", "yellow", "black",
                  "white", "gray", "brown","orange", "pink", "purple"]
    color_count = sum(1 for color in color_bank if color in response_text)
    assert color_count>=1, "response text should have atleast 1 color"
    print("Colors named:" , response_text)
    print("Count of colors from bank named:", color_count)

def test_simple_qa(llm_client):
    response = llm_client.generate("Whats the capital of Italy?")
    assert "rome" in response['text'].lower()
    print(response['text'])

def test_multi_turn_basic(llm_client):
    prompt = """
    User: Hi my name is Humera
    Assistant: Hi Humera, how can i help you?
    User: What is my name?"""
    response = llm_client.generate(prompt)

    assert "humera" in response['text'].lower()
    print(response['text'])

