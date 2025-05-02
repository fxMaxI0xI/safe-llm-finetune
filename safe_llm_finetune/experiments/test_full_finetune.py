import pandas as pd
from pathlib import Path
from safe_llm_finetune.fine_tuning.models.gemma_3_1B_it_adapter import GemmaAdapter
from safe_llm_finetune.fine_tuning.methods.full_fine_tuning import FullFineTuning
from safe_llm_finetune.fine_tuning.base import TrainingConfig, CheckpointConfig

# Dummy-Daten im Alpaca-Stil
data = pd.DataFrame([
    {"prompt": "Schreibe eine Python-Funktion, die zwei Zahlen addiert.", "completion": "def addiere(a, b):\n    return a + b"},
    {"prompt": "Wie lautet die Hauptstadt von Deutschland?", "completion": "Berlin."}
])

# Modelladapter + Finetuning-Klasse initialisieren
adapter = GemmaAdapter()
method = FullFineTuning(adapter)

# Modell + Tokenizer laden
model_id = "google/gemma-3-1b-it"
model = adapter.load_model(model_id)
tokenizer = adapter.load_tokenizer(model_id)

# Dummy-Config für schnelles Testtraining
config = TrainingConfig(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    fp16=False,
    checkpoint_config=CheckpointConfig(
        checkpoint_dir=Path("./test_output")
    )
)

# Dataset vorbereiten und Training starten
tokenized_dataset = method.prepare_dataset(data)
model = method.train(model, tokenizer, tokenized_dataset, config)

# Modell lokal speichern
adapter.save_model(model, "./test_output")
