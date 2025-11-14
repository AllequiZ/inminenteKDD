import pandas as pd
from sklearn.decomposition import PCA
import umap
import plotly.express as px

# === CONFIGURAÇÕES ===
CAMINHO_CSV = "fragmentos_clusters.csv"
USA_UMAP = True  # True = UMAP (melhor visualização), False = PCA
N_COMPONENTS = 2
ARQUIVO_SAIDA = "clusters_mapa.csv"

print("📂 Carregando dataset...")
df = pd.read_csv(CAMINHO_CSV)

# === 1. Extrair as colunas numéricas ===
dim_cols = [c for c in df.columns if c.startswith("dim_")]
X = df[dim_cols].values

# === 2. Reduzir dimensionalidade ===
print("🔍 Reduzindo dimensionalidade...")

if USA_UMAP:
    reducer = umap.UMAP(n_components=N_COMPONENTS, random_state=42)
    metodo = "UMAP"
else:
    reducer = PCA(n_components=N_COMPONENTS)
    metodo = "PCA"

X_2d = reducer.fit_transform(X)
df["x"] = X_2d[:, 0]
df["y"] = X_2d[:, 1]

print(f"✅ Redução concluída usando {metodo}.")

# === 3. Visualização interativa ===
print("🎨 Gerando mapa interativo...")

fig = px.scatter(
    df,
    x="x",
    y="y",
    color="cluster",
    hover_data=["titulo", "texto_fragmento"],
    title=f"Mapa de Temas do Podcast ({metodo})",
    color_continuous_scale="Viridis",
    opacity=0.7,
    width=900,
    height=700
)

fig.update_traces(marker=dict(size=6))
fig.show()

# === 4. Exportar CSV com coordenadas 2D ===
df_saida = df[["id", "titulo", "fragmento_id", "texto_fragmento", "cluster", "x", "y"]]
df_saida.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8")
print(f"💾 Arquivo salvo em: {ARQUIVO_SAIDA}")

print("🎯 Interpretação concluída!")
