# MetaMind: LLM-Orchestrated Computational Intelligence Framework

An LLM-based intelligent agent that orchestrates various Computational Intelligence (CI) methods to solve optimization, classification, and clustering problems.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python -m src.main
```

## Project Structure

- `src/orchestrator/` - LLM orchestration components
- `src/methods/` - CI method implementations (9 methods)
- `src/problems/` - Benchmark problem definitions
- `src/evaluation/` - Metrics and visualization
- `experiments/` - Experiment scripts
