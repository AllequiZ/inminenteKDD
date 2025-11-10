import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# === INTENTO DE IMPORTAR HDBSCAN ===
HAS_HDBSCAN = True
try:
    import hdbscan
except Exception as e:
    HAS_HDBSCAN = False

# === CONFIG ===
CAMINHO_VETORES = "vetores_fragmentados_bert.csv"
MODO_PREF = "hdbscan"   # "hdbscan" o "kmeans"
NUM_CLUSTERS = 8        # usado solo para KMeans
TOP_N_SIMILARES = 3
TOP_EXEMPLOS_POR_CLUSTER = 6
SALVAR_RESULTADOS = True

# === 1. LECTURA ===
df = pd.read_csv(CAMINHO_VETORES)
print(f"📂 Dataset cargado: {len(df)} fragmentos")

# columnas de vectores (dim_*)
vetores = df.filter(regex="^dim_").values
if vetores.size == 0:
    raise RuntimeError("No se encontraron columnas 'dim_' en el CSV. Asegúrate de usar el CSV generado por la vectorización fragmentada.")

# === 2. CLUSTERING ===
if MODO_PREF == "hdbscan":
    if HAS_HDBSCAN:
        print("🔹 Usando HDBSCAN (ideal para temas variados)...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
        clusters = clusterer.fit_predict(vetores)
    else:
        print("⚠️ HDBSCAN no está instalado. Usando KMeans en su lugar.")
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
        clusters = kmeans.fit_predict(vetores)
elif MODO_PREF == "kmeans":
    print("🔹 Usando KMeans...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
    clusters = kmeans.fit_predict(vetores)
else:
    raise ValueError("MODO_PREF inválido. Usa 'hdbscan' o 'kmeans'.")

df["cluster"] = clusters
print(f"✅ Clustering terminado. Clusters únicos: {np.unique(clusters)}")

# === 3. SIMILARIDAD COSENO ===
print("🧭 Calculando similaridad coseno entre fragmentos...")
sim = cosine_similarity(vetores)

fragmentos_similares = []
for i, row_i in df.iterrows():
    idxs = np.argsort(sim[i])[::-1]
    idxs = idxs[idxs != i][:TOP_N_SIMILARES]
    for j in idxs:
        row_j = df.iloc[j]
        fragmentos_similares.append({
            "id_ep_referencia": row_i["id"],
            "titulo_referencia": row_i["titulo"],
            "fragmento_id_ref": row_i["fragmento_id"],
            "id_ep_similar": row_j["id"],
            "titulo_similar": row_j["titulo"],
            "fragmento_id_sim": row_j["fragmento_id"],
            "similaridade": float(sim[i][j])
        })

similares_df = pd.DataFrame(fragmentos_similares)

# === 4. GUARDAR RESULTADOS ===
if SALVAR_RESULTADOS:
    df.to_csv("fragmentos_clusters.csv", index=False)
    similares_df.to_csv("fragmentos_similares.csv", index=False)
    print("💾 Guardados: fragmentos_clusters.csv, fragmentos_similares.csv")

# === 5. EJEMPLOS POR CLUSTER ===
texto_col = "texto_fragmento" if "texto_fragmento" in df.columns else "titulo"

cluster_examples = []
for c in sorted(df["cluster"].unique()):
    dfc = df[df["cluster"] == c]
    sample = dfc.head(TOP_EXEMPLOS_POR_CLUSTER)
    for _, row in sample.iterrows():
        cluster_examples.append({
            "cluster": int(c),
            "id_ep": row["id"],
            "fragmento_id": row["fragmento_id"],
            "titulo": row["titulo"],
            "texto_ejemplo": row[texto_col][:250] + "..."  # recorte para legibilidad
        })

examples_df = pd.DataFrame(cluster_examples)
examples_df.to_csv("cluster_fragmentos_ejemplos.csv", index=False)
print("💾 Guardado: cluster_fragmentos_ejemplos.csv (ejemplos por cluster para interpretar)")

print("🎯 Minería finalizada.")
if not HAS_HDBSCAN and MODO_PREF == "hdbscan":
    print("\nℹ️ Nota: Para usar HDBSCAN instala con:\n   pip install hdbscan\n   o\n   conda install -c conda-forge hdbscan")
