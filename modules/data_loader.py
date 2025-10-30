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
