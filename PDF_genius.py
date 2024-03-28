from transformers import T5ForConditionalGeneration, T5Tokenizer
import fitz  # PyMuPDF

# Charger le modèle et le tokenizer T5 pré-entraîné
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")


def lire_texte_pdf(nom_fichier):
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

    return texte_complet


def repondre_question_pdf(question, texte_pdf):
    # Formuler la question sous forme de prompt
    prompt = f"question: {question} contexte: {texte_pdf}"

    # Tokenization et encodage du prompt pour le modèle
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    # Générer la réponse avec le modèle
    réponse_ids = model.generate(inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4,
                                 early_stopping=True)
    réponse = tokenizer.decode(réponse_ids[0], skip_special_tokens=True)

    return réponse


# Utilisation de la fonction pour lire le texte du fichier PDF
nom_fichier_pdf = "Rapport_Alerte_Ingerence_Fondations.pdf"  # Remplacez ceci par le chemin de votre fichier PDF
texte_pdf = lire_texte_pdf(nom_fichier_pdf)

# Poser une question sur le contenu du PDF
question = "Quels sont les principaux points abordés dans ce document ?"
réponse = repondre_question_pdf(question, texte_pdf)
print("Réponse du modèle T5 à la question:", réponse)
