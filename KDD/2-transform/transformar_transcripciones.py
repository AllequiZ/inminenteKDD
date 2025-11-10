import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np

# === CONFIGURAÇÕES ===
CAMINHO_CSV = "dataset_podcast.csv"
MODO = "bert"  # pode ser "tfidf" ou "bert"
SALVAR_EMBEDS = True  # se True, salva os vetores resultantes

# === 1. LEITURA DO DATASET ===
df = pd.read_csv(CAMINHO_CSV)

# Se o campo descricao_limpa já contém todo o texto, usamos ele:
textos = df["transcripcion_limpia"].fillna("").tolist()

# === 2. VETORIZAÇÃO ===
if MODO == "tfidf":
    print("🧮 Aplicando TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='portuguese', max_features=5000)
    matriz = vectorizer.fit_transform(textos)
    feature_names = vectorizer.get_feature_names_out()
    matriz_array = matriz.toarray()

elif MODO == "bert":
    print("🤖 Gerando embeddings BERT...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # leve e eficiente
    matriz_array = model.encode(textos, show_progress_bar=True)
    feature_names = [f"dim_{i}" for i in range(matriz_array.shape[1])]

else:
    raise ValueError("Modo inválido. Use 'tfidf' ou 'bert'.")

# === 3. ORGANIZAÇÃO DOS RESULTADOS ===
matriz_df = pd.DataFrame(matriz_array, columns=feature_names)
resultado = pd.concat([df[["id", "titulo"]], matriz_df], axis=1)

# === 4. SALVAR (opcional) ===
if SALVAR_EMBEDS:
    saida = "vetores_podcast_" + MODO + ".csv"
    resultado.to_csv(saida, index=False)
    print(f"✅ Vetores salvos em: {saida}")

print("🎯 Vetorização concluída!")
