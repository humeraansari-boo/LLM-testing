# LLM Testing

A hands-on project for learning how to test Large Language Models (LLMs) with Python and pytest.

The project tests LLMs in two ways:

1. **Native tests**: plain pytest tests that check model output using string matching, regex and simple metrics. They run against a local model served by [Ollama](https://ollama.com).
2. **Framework tests** (in progress): tests built with [DeepEval](https://github.com/confident-ai/deepeval), an LLM evaluation framework that uses an "LLM-as-a-judge" to score qualities like relevancy, correctness and hallucination.

Comparing the two shows what hand-written assertions can catch and where a dedicated evaluation framework adds value.

---

## Project structure

```
LLM testing/
├── llamatestclient.py          # Client for a local Ollama model + pytest fixture
├── openaitestclient.py         # Client for OpenAI models (planned)
├── pytest.ini                  # pytest configuration
├── .env                        # API keys (local only, NOT committed)
└── tests/
    ├── native/                 # Hand-written tests against the local model
    │   ├── test_basic_functionality.py
    │   ├── test_context_learning.py
    │   ├── test_context_processing_generator.py
    │   ├── test_hallucination_detections.py
    │   ├── test_performance.py
    │   └── test_robustness.py
    └── framework/              # DeepEval-based tests (in progress)
        └── test_basic_functionality.py
```

---

## How it works

### The test client: `llamatestclient.py`

`OllamaTestClient` sends prompts to Ollama's local REST API (`http://localhost:11434/api/generate`) and returns a dictionary with:

| Key                 | Meaning                                         |
|---------------------|-------------------------------------------------|
| `text`              | The model's response                            |
| `prompt_tokens`     | Approximate prompt length (word count)          |
| `completion_tokens` | Approximate response length (word count)        |
| `latency`           | Time taken for the request, in seconds          |

The `llm_client` pytest fixture creates a client for the `tinyllama` model. If the model isn't available, the tests are **skipped** rather than failed.

### Native test suites (`tests/native/`)

| File | What it tests |
|------|---------------|
| `test_basic_functionality.py` | The model responds with non-empty text, follows a simple instruction ("name 3 colors"), answers a factual question (capital of Italy → Rome) and remembers context in a multi-turn conversation. |
| `test_context_learning.py` | **Logical consistency**: two equivalent questions ("What is the capital of Italy?" / "Rome is the capital of which country?") should give overlapping answers, measured by word overlap (Jaccard similarity ≥ 0.2). |
| `test_context_processing_generator.py` | **Output formatting**: the model can produce a numbered list and a basic JSON object when asked. |
| `test_hallucination_detections.py` | **Hallucination**: for impossible questions ("What's the capital of Barbieland?"), the model shouldn't make up a confident, specific answer. |
| `test_performance.py` | **Performance**: sends repeated requests and reports total time, tokens per second and requests per minute. |
| `test_robustness.py` | **Robustness to input variation**: the same question phrased differently (casing, wording, typos) should still produce the correct answer. |

### Framework tests (`tests/framework/`)

This folder is for tests written with **DeepEval**. DeepEval provides ready-made metrics such as:

- `AnswerRelevancyMetric`: is the answer relevant to the question?
- `HallucinationMetric`: does the answer contradict the given context?
- `GEval`: custom criteria scored by an LLM judge

By default DeepEval uses an OpenAI model as the judge, which is why an OpenAI API key is needed (see setup below). It can also be configured to use a local Ollama model instead.

---

## Setup

### Prerequisites

- Python 3.10+ (developed with Python 3.14)
- [Ollama](https://ollama.com/download) installed and running
- An OpenAI API key (only for the DeepEval tests)

### 1. Clone the repository

```powershell
git clone https://github.com/humeraansari-boo/LLM-testing.git
cd LLM-testing
```

### 2. Create a virtual environment and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pytest requests deepeval
```

### 3. Download the local model

```powershell
ollama pull tinyllama
```

Make sure Ollama is running (it serves on `localhost:11434` by default).

### 4. Add your OpenAI API key

Create a file named `.env` in the project root:

```
OPENAI_API_KEY=your-key-here
```

DeepEval loads this file automatically. `.env` is listed in `.gitignore`, so your key is never committed.

---

## Running the tests

Run everything:

```powershell
pytest
```

Run only the native tests, showing printed output:

```powershell
pytest tests/native -s
```

Run a single test file:

```powershell
pytest tests/native/test_robustness.py -v
```

Run the DeepEval tests:

```powershell
deepeval test run tests/framework/
```

---

## Branches

| Branch | Purpose |
|--------|---------|
| `master` | Native pytest tests against a local Ollama model |
| `deep-eval-testing` | Adding DeepEval-based framework tests |

---

## Roadmap

- [x] Ollama test client and pytest fixture
- [x] Native tests: functionality, consistency, formatting, hallucination, performance, robustness
- [x] Install DeepEval and configure the API key
- [ ] OpenAI test client (`openaitestclient.py`)
- [ ] DeepEval tests in `tests/framework/`
- [ ] Compare native vs. DeepEval results
