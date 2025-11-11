# Importación de librerías y supresión de advertencias
from os.path import join # Solves path issues across OS
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport # Librería para reportes EDA

def visualizar_datos(fuente: str = join("data", "raw", "bank-additional-full.csv"),
                    salida: str = join("docs", "figures")) -> None:
    """ Genera una serie de gráficos sobre los datos de fuente y los guarda en la ruta salida.

    Parámetros:
        fuente (str, opcional): Ruta al archivo CSV con los datos.
        salida (str, opcional): Ruta donde se guardarán los gráficos generados.
    """
    # Crear el directorio de salida si no existe
    Path(salida).mkdir(parents=True, exist_ok=True)

    # Se cargan los datos
    df = pd.read_csv(fuente, sep=';')

    # Gráfico 1: Distribución de la variable objetivo 'y'
    print("Generando gráfico 1: Distribución de la variable objetivo 'y'")
    plt.figure(figsize=(6, 4))
    sns.countplot(x="y", data=df)
    plt.title("Distribución de la variable objetivo (suscripción al depósito)")
    plt.xlabel("¿Suscribió un depósito a plazo?")
    plt.ylabel("Cantidad de clientes")
    plt.savefig(join(salida, 'grafico01_distribucion_y.png'))
    print("Gráfico 1 guardado en:", 
          join(salida, 'grafico01_distribucion_y.png'))
    plt.close()

    df["y"].value_counts(normalize=True).mul(100).round(2)

    # Gráfico 2: Distribución del nivel de educación
    print("Generando gráfico 2: Distribución del nivel de educación")
    col = 'education'
    plt.figure(figsize=(6, 4))
    order = df[col].value_counts().index
    sns.countplot(y=col, data=df, order=order)
    plt.title(f"Distribución de {col}")
    plt.xlabel("Cantidad")
    plt.ylabel(col)
    plt.savefig(join(salida, 'grafico02_distribucion_educacion.png'))
    print("Gráfico 2 guardado en:", 
          join(salida, 'grafico02_distribucion_educacion.png'))
    plt.close()

    # Gráfico 3: Distribución de ocupación laboral
    print("Generando gráfico 3: Distribución de ocupación laboral")
    col = 'job'
    plt.figure(figsize=(6, 4))
    order = df[col].value_counts().index
    sns.countplot(y=col, data=df, order=order)
    plt.title(f"Distribución de {col}")
    plt.xlabel("Cantidad")
    plt.ylabel(col)
    plt.savefig(join(salida, 'grafico03_distribucion_ocupacion.png'))
    print("Gráfico 3 guardado en:", 
          join(salida, 'grafico03_distribucion_ocupacion.png'))
    plt.close()

    # Gráfico 4: Distribución de préstamos personales
    print("Generando gráfico 4: Distribución de préstamos personales")
    col = 'loan'
    plt.figure(figsize=(6, 4))
    order = df[col].value_counts().index
    sns.countplot(y=col, data=df, order=order)
    plt.title(f"Distribución de {col}")
    plt.xlabel("Cantidad")
    plt.ylabel(col)
    plt.savefig(join(salida, 'grafico04_distribucion_prestamospersonales.png'))
    print("Gráfico 4 guardado en:", 
          join(salida, 'grafico04_distribucion_prestamospersonales.png'))
    plt.close()

    # Gráfico 5: Distribución de préstamos hipotecarios
    print("Generando gráfico 5: Distribución de préstamos hipotecarios")
    col = 'housing'
    plt.figure(figsize=(6, 4))
    order = df[col].value_counts().index
    sns.countplot(y=col, data=df, order=order)
    plt.title(f"Distribución de {col}")
    plt.xlabel("Cantidad")
    plt.ylabel(col)
    plt.savefig(join(salida, 'grafico05_distribucion_prestamoshipotecarios.png'))
    print("Gráfico 5 guardado en:", 
          join(salida, 'grafico05_distribucion_prestamoshipotecarios.png'))
    plt.close()

    # Gráfico 6: Matriz de correlación
    print("Generando gráfico 6: Matriz de correlación")
    num_df = df.select_dtypes(include=['float64', 'int64'])
    corr = num_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de correlaciones')
    plt.savefig(join(salida, 'grafico06_matrizcorrelacion.png'))
    print("Gráfico 6 guardado en:", 
          join(salida, 'grafico06_matrizcorrelacion.png'))
    plt.close()

def generacion_reporte_eda(fuente: str = join("data", "raw", "bank-additional-full.csv"),
                    salida: str = join("docs", "reports")) -> None:

    """Genera un reporte con la librería ydata_profiling sobre los datos de fuente 
    y lo guarda en la ruta salida.
    
    Parámetros:
        fuente (str, opcional): Ruta al archivo CSV con los datos.
        salida (str, opcional): Ruta donde se guardará el reporte generado.
    """

    # Crear el directorio de salida si no existe
    Path(salida).mkdir(parents=True, exist_ok=True)
    # Se cargan los datos
    df = pd.read_csv(fuente, sep=';')
    # Generar el reporte
    profile = ProfileReport(df, title="Reporte de Datos", explorative=True)
    # Guardar el reporte en un archivo HTML
    profile.to_file(join(salida, "EDA_reporte.html"))

if __name__ == "__main__":
    visualizar_datos()
    generacion_reporte_eda()
