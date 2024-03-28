from transformers import pipeline

import logging
logging.basicConfig(level=logging.INFO)

# Utilise un modèle plus léger, DistilBERT
fill_mask = pipeline("fill-mask", model="distilbert-base-uncased")
resultats = fill_mask("The capital of France is [MASK].")
print(resultats)
