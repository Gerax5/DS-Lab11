import pandas as pd

def load_excel_data(path: str) -> dict:
    xls = pd.read_excel(path, sheet_name=None)
    resultados = {}

    for nombre_hoja, _ in xls.items():
        df = pd.read_excel(
            path, 
            sheet_name=nombre_hoja, 
            header=7, 
            usecols="A,C,D,E,F,G,H"
        )
        df = df.iloc[1:].reset_index(drop=True)
        resultados[nombre_hoja] = df

    return resultados


def load_consumption_excel_data() -> dict:
    hoja_df = pd.read_excel("./CONSUMO-HIDROCARBUROS-2024-12.xlsx", header=6)

    hoja_df = hoja_df.iloc[:-3]

    hoja_df["Fecha"] = pd.to_datetime(hoja_df["Fecha"])
    hoja_df["Año"] = hoja_df["Fecha"].dt.year
    hoja_df["Mes"] = hoja_df["Fecha"].dt.month

# Select just necesary columns
    gas_df = hoja_df[["Fecha", "Gas licuado de petróleo", "Año", "Mes"]]
    gas_df = gas_df.set_index('Fecha').sort_index()
    return hoja_df