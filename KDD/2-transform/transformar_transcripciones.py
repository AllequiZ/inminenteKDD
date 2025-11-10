import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
import textwrap

# === CONFIGURACIONES ===
CAMINHO_CSV = "dataset_podcast.csv"
MODO = "bert"  # "tfidf" o "bert"
SALVAR_EMBEDS = True
PALAVRAS_POR_FRAGMENTO = 150  # ajusta según lo largo de tus textos

# === 1. LEER DATASET ===
df = pd.read_csv(CAMINHO_CSV)
df["transcripcion_limpia"] = df["transcripcion_limpia"].fillna("")

# === 2. DIVISIÓN EN FRAGMENTOS ===
print("🧩 Dividiendo transcripciones en fragmentos...")

fragmentos = []
for _, row in df.iterrows():
    id_ep = row["id"]
    titulo = row["titulo"]
    texto = row["transcripcion_limpia"]

    palabras = texto.split()
    num_fragmentos = max(1, len(palabras) // PALAVRAS_POR_FRAGMENTO + (1 if len(palabras) % PALAVRAS_POR_FRAGMENTO != 0 else 0))

    for i in range(num_fragmentos):
        inicio = i * PALAVRAS_POR_FRAGMENTO
        fin = inicio + PALAVRAS_POR_FRAGMENTO
        fragmento_texto = " ".join(palabras[inicio:fin]).strip()
        if fragmento_texto:
            fragmentos.append({
                "id": id_ep,
                "titulo": titulo,
                "fragmento_id": i + 1,
                "texto_fragmento": fragmento_texto
            })

fragmentos_df = pd.DataFrame(fragmentos)
print(f"📚 Total de fragmentos generados: {len(fragmentos_df)}")

# === 3. VECTORIZACIÓN ===
if MODO == "tfidf":
    print("🧮 Aplicando TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='portuguese', max_features=5000)
    matriz = vectorizer.fit_transform(fragmentos_df["texto_fragmento"])
    feature_names = vectorizer.get_feature_names_out()
    matriz_array = matriz.toarray()

elif MODO == "bert":
    print("🤖 Generando embeddings BERT (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    matriz_array = model.encode(fragmentos_df["texto_fragmento"].tolist(), show_progress_bar=True)
    feature_names = [f"dim_{i}" for i in range(matriz_array.shape[1])]

else:
    raise ValueError("Modo inválido. Use 'tfidf' o 'bert'.")

# === 4. COMBINAR Y GUARDAR ===
matriz_df = pd.DataFrame(matriz_array, columns=feature_names)
resultado = pd.concat([fragmentos_df, matriz_df], axis=1)

if SALVAR_EMBEDS:
    saida = f"vetores_fragmentados_{MODO}.csv"
    resultado.to_csv(saida, index=False)
    print(f"✅ Vetores salvos em: {saida}")

print("🎯 Vectorización concluida con éxito.")

