import matplotlib.pyplot as plt
import pandas as pd

def plot_price_distribution(df_final):
    # Asegurar estilo base limpio
    plt.style.use("default")

    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(10, 5))

    if "Superior" in df_final.columns:
        df_final["Superior"].dropna().plot(
            kind="hist",
            bins=20,
            color="#f46530",      
            edgecolor="black",
            ax=ax
        )

        ax.set_facecolor("#2a2e30")          
        fig.patch.set_facecolor("#2a2e30")  

        ax.set_title("Distribución total de precios de Gasolina Superior", fontsize=22, color="white")
        ax.set_xlabel("Precio (Q)", fontsize=20, color="white")
        ax.set_ylabel("Frecuencia", fontsize=20, color="white")

        ax.tick_params(colors="white", labelsize=16)
        for spine in ax.spines.values():
            spine.set_color("white")

        ax.grid(True, color="gray", alpha=0.4)
    else:
        ax.text(
            0.5, 0.5,
            "No se encontró la columna 'Superior' en el DataFrame",
            color="white",
            ha="center", va="center",
            fontsize=16
        )
        ax.set_facecolor("#2a2e30")
        fig.patch.set_facecolor("#2a2e30")

    return fig


def plot_monthly_avg_price(df):

    if "Superior" not in df.columns:
        raise ValueError("El DataFrame no contiene la columna 'Superior'")

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            raise ValueError("No se pudo convertir el índice a fechas.")

    df = df.dropna(subset=[df.index.name]) if df.index.name else df

    df["Mes"] = df.index.month
    monthly_mean = df.groupby("Mes")["Superior"].mean()

    best_month = monthly_mean.idxmin()
    best_price = monthly_mean.min()

    meses = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_mean.plot(
        kind="bar",
        color="#f46530",
        edgecolor="#ff9e7a",
        ax=ax,
    )

    ax.set_facecolor("#2a2e30")
    fig.patch.set_facecolor("#2a2e30")

    ax.set_title("Precio promedio mensual de gasolina Superior", fontsize=23, color="white")
    ax.set_xlabel("Mes", fontsize=19, color="white")
    ax.set_ylabel("Precio promedio", fontsize=19, color="white")

    ax.set_xticks(range(len(meses)))
    ax.set_xticklabels(meses, rotation=45, ha="right", color="white", fontsize=17)

    ax.tick_params(colors="white", labelsize=16)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.bar(
        best_month - 1,
        best_price,
        color="#ff9e7a",
        edgecolor="black",
        width=0.55,
    )

    ax.grid(False)

    plt.tight_layout()
    return fig