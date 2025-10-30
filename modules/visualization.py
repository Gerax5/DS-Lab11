import matplotlib.pyplot as plt

def plot_price_distribution(df_final):
    # Asegurar estilo base limpio
    plt.style.use("default")

    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(10, 5))

    # --- Histograma ---
    if "Superior" in df_final.columns:
        df_final["Superior"].dropna().plot(
            kind="hist",
            bins=20,
            color="#f46530",      # 🔶 barras naranja-rojas
            edgecolor="black",
            ax=ax
        )

        # --- Fondo del gráfico ---
        ax.set_facecolor("#2a2e30")          # Fondo interno
        fig.patch.set_facecolor("#2a2e30")   # Fondo externo

        # --- Títulos y etiquetas ---
        ax.set_title("Distribución total de precios de Gasolina Superior", fontsize=22, color="white")
        ax.set_xlabel("Precio (Q)", fontsize=20, color="white")
        ax.set_ylabel("Frecuencia", fontsize=20, color="white")

        # --- Ejes en blanco ---
        ax.tick_params(colors="white", labelsize=16)
        for spine in ax.spines.values():
            spine.set_color("white")

        # --- Cuadrícula tenue ---
        ax.grid(True, color="gray", alpha=0.4)
    else:
        # Mensaje si no existe la columna 'Superior'
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
