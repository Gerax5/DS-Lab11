import pandas as pd

def load_excel_data(path: str) -> dict:
    """
    Load price data from Excel file with multiple sheets.
    Returns a dictionary with year as key and dataframe as value.
    """
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
        
        # Rename columns
        df.rename(columns={'Glp Cilindro 25Lbs.': 'Gas Licuado'}, inplace=True)
        
        # Set date as index
        if 'Fecha' in df.columns:
            df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
            df = df.set_index('Fecha')
        elif df.columns[0] not in ['Superior', 'Regular', 'Diesel', 'Gas Licuado']:
            # If first column is not a product, assume it's the date
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
            df = df.set_index(df.columns[0])
        
        resultados[nombre_hoja] = df

    return resultados


def load_consumption_excel_data() -> pd.DataFrame:
    """
    Load consumption data from Excel file.
    Returns a dataframe with datetime index and consumption columns including Year and Month.
    
    DataFrame structure:
                        Gas licuado de petróleo  Año  Mes
    Fecha                                                  
    2000-01-01                     194410.476190  2000    1
    2000-02-01                     174710.552381  2000    2
    2000-03-01                     189234.066667  2000    3
    """
    # Load the Excel file
    hoja_df = pd.read_excel("./data/CONSUMO-HIDROCARBUROS-2024-12.xlsx", header=6)
    
    # Remove last 3 rows (usually summary rows)
    hoja_df = hoja_df.iloc[:-3]
    
    # Process date column
    hoja_df["Fecha"] = pd.to_datetime(hoja_df["Fecha"], errors='coerce')
    
    # Add Year and Month columns
    hoja_df["Año"] = hoja_df["Fecha"].dt.year
    hoja_df["Mes"] = hoja_df["Fecha"].dt.month
    
    # Select necessary columns (modify this based on your actual column names)
    # The consumption column might have a different name in your file
    # Common names: "Gas licuado de petróleo", "GLP", "Gas Licuado", etc.
    gas_df = hoja_df[["Fecha", "Gas licuado de petróleo", "Año", "Mes"]].copy()
    
    # Set date as index and sort
    gas_df = gas_df.set_index('Fecha').sort_index()
    
    # Remove any rows with NaT index
    gas_df = gas_df[gas_df.index.notna()]
    
    print(gas_df.head(10))
    
    return gas_df