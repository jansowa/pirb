# 1. Prepare environment
```shell
pip install uv
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.lock
uv pip install flash-attn==2.8.2 --no-build-isolation
python run_benchmark.py --scope tiny --models_config config/mmlw/mmlw-retrieval-roberta-base.json --benchmark_config "config/benchmarks/pirb-without-private.json"
```

# 2. Script execution example:
```shell
python run_benchmark.py --scope tiny --models_config config/mmlw/mmlw-retrieval-roberta-large-v2.json --benchmark_config "config/benchmarks/pirb-without-private.json"
```

You can use `full`/`small`/`tiny` scope (all/<100MB/<20MB datasets). Prepare model config for `models_config` parameter.