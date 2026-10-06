from llamatestclient import llm_client


def test_logical_consistency(llm_client):
    equivalent_questions = [("Whats is the capital of Italy?", "Rome is the capital of which country?")]

    for q1,q2 in equivalent_questions:
        response1 = llm_client.generate(q1)
        response2 = llm_client.generate(q2)

        key_info1 = response1["text"].lower()[:100]
        key_info2 = response2["text"].lower()[:100]

        words1 = set(key_info1.split())
        words2 = set(key_info2.split())
        print(response1["text"])
        print(response2["text"])

        if words1 and words2:
            common = words1.intersection(words2)
            union = words1.union(words2)
            overlap_ratio =len(common)/len(union)
            assert overlap_ratio >= 0.2, "inconsistent answers for equivalent questions"

