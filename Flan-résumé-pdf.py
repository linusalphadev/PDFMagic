
from transformers import T5ForConditionalGeneration, T5Tokenizer  # pip install PyMuPDF transformers

import fitz  # PyMuPDF

# Charger le modèle et le tokenizer T5 pré-entraîné
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def lire_et_resumer_pdf(nom_fichier):
    # Ouvrir le fichier PDF
    document = fitz.open(nom_fichier)
    texte_complet = ""

    # Concaténer le texte de toutes les pages
    for num_page in range(len(document)):
        page = document[num_page]
        texte_page = page.get_text()
        texte_complet += texte_page

    # Fermer le document
    document.close()

    # Prétraiter le texte pour le résumé
    texte_pretraite = "summarize: " + texte_complet

    # Tokenization et encodage du texte pour le modèle
    inputs = tokenizer(texte_pretraite, return_tensors="pt", max_length=512, truncation=True)

    # Générer le résumé avec le modèle
    résumé_ids = model.generate(inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    résumé = tokenizer.decode(résumé_ids[0], skip_special_tokens=True)

    return résumé

# Utilisation de la fonction pour lire et résumer un fichier PDF
nom_fichier_pdf = "Rapport_Alerte_Ingerence_Fondations.pdf"  # Remplacez ceci par le chemin de votre fichier PDF
résumé = lire_et_resumer_pdf(nom_fichier_pdf)
print("Résumé du PDF:")
print(résumé)
