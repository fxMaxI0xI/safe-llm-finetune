from pathlib import Path
import sys
# Add project root to Python path to enable imports from the package
sys.path.append(str(Path.cwd().parent))

from safe_llm_finetune.evaluation.codalbench import CodalBench
import os
from huggingface_hub import login



HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

codal = CodalBench( debug = True)

log = codal.run_eval(model_path="...", tokenizer_path="...", base_path="test")

