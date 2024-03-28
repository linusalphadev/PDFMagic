import fitz  # PyMuPDF


def extraire_texte_de_pdf(chemin_pdf):
    document = fitz.open(chemin_pdf)
    texte = ""
    for page in document:
        texte += page.get_text()
    return texte


from transformers import pipeline

# Utilise un modèle plus léger, DistilBERT
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased")


def repondre_a_la_question(question, contexte):
    result = qa_pipeline(question=question, context=contexte)
    return result['answer']


chemin_pdf = "PDFfolder/OSINT veille et intelligence économique.pdf"
texte_pdf = extraire_texte_de_pdf(chemin_pdf)

question = "Quel est le sujet de ce document"
reponse = repondre_a_la_question(question, texte_pdf)

print(f"Question: {question}")
print(f"Réponse: {reponse}")
