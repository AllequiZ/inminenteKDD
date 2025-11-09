import re
import pandas as pd
from pathlib import Path

# --- CONFIGURACIÓN ---
CARPETA_TRANSCRIPCIONES = Path("dados_podcast/")   # Carpeta con los .txt o .vtt
DATASET_PATH = Path("dataset_podcast.csv")       # Archivo donde se guardan los resultados


def limpiar_texto(texto: str) -> str:
    """Elimina timestamps, saltos de línea, múltiples espacios y normaliza el texto."""
    # Quitar líneas con tiempos (ej: 00:00:00.000 --> 00:00:02.859)
    texto = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3}\s-->\s\d{2}:\d{2}:\d{2}\.\d{3}", "", texto)
    # Quitar líneas vacías y espacios múltiples
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def cargar_dataset():
    """Carga el dataset existente o crea uno vacío."""
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    else:
        return pd.DataFrame(columns=["id", "titulo", "archivo", "transcripcion_limpia"])


def procesar_archivos():
    df = cargar_dataset()

    # Archivos ya procesados
    archivos_existentes = set(df["archivo"].tolist())

    # Buscar todos los .txt o .vtt en la carpeta
    nuevos_archivos = [f for f in CARPETA_TRANSCRIPCIONES.glob("*.*") if f.suffix in [".txt", ".vtt"] and f.name not in archivos_existentes]

    if not nuevos_archivos:
        print("✅ No hay archivos nuevos por procesar.")
        return

    nuevos_registros = []

    for archivo in nuevos_archivos:
        print(f"\n📄 Procesando archivo: {archivo.name}")

        # Leer contenido
        with open(archivo, "r", encoding="utf-8") as f:
            texto = f.read()

        # Limpiar
        texto_limpio = limpiar_texto(texto)

        # Pedir datos manuales
        id_ep = input("🆔 ID del episodio (ej: ep012): ").strip()
        titulo_ep = input("🎙️ Título del episodio: ").strip()

        nuevos_registros.append({
            "id": id_ep,
            "titulo": titulo_ep,
            "archivo": archivo.name,
            "transcripcion_limpia": texto_limpio
        })

    # Agregar al dataset existente
    df = pd.concat([df, pd.DataFrame(nuevos_registros)], ignore_index=True)

    # Guardar
    df.to_csv(DATASET_PATH, index=False, encoding="utf-8")
    print(f"\n✅ {len(nuevos_registros)} nuevos episodios agregados al dataset: {DATASET_PATH.name}")


if __name__ == "__main__":
    procesar_archivos()
