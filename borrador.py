import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew ,norm,t
import scipy.stats as stats 
from scipy.stats import t

#FUNCIONES 

def obtener_datos(stocks):
    '''
    El objetivo de esta funcion es descargar el precio
    de cierre de un o varios activos desde el 2010
    '''
    df = yf.download(stocks, start="2010-01-01", end=datetime.now().strftime('%Y-%m-%d'))['Close']
    return df

def calcular_rendimientos(df):
    '''
    Calcula los rendimientos de un activo
    '''
    return df.pct_change().dropna()

def calcular_rendimientos_log(df):
    '''
    Funcion que calcula los rendimientos logarítmicos de un activo.
    '''
    return np.log(df / df.shift(1)).dropna()

def calcular_metricas(df):
    '''
    Funcion que Determina la variación porcentual diaria entre los precios,
    calcula los retornos acumulados y normaliza los precios.
    '''
    returns = df.pct_change().dropna()
    cumulative_returns = (1 + returns).cumprod() - 1
    normalized_prices = df / df.iloc[0] * 100
    return returns, cumulative_returns, normalized_prices

def calcular_var(returns, alpha=0.95):
    """
    Calcula el VaR histórico como el percentil empírico de los rendimientos.
    """
    return np.percentile(returns, (1 - alpha) * 100)

def calcular_cvar(returns, alpha=0.95):
    """
    Calcula el CVaR (Expected Shortfall) usando datos históricos.
    """
    var = calcular_var(returns, alpha)
    cvar = returns[returns <= var].mean()  # Promedio de pérdidas más extremas
    return cvar



# Configuración de la página de Streamlit
st.set_page_config(page_title="Metricas de acciones", layout="wide")

# Crear pestañas
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Metricas básicas y rendimientos", "Var & cVaR", "Rolling Windows", "Violaciones", "VaR Volatilidad Móvil"])
# Título en la barra lateral
st.sidebar.title("Analizador de Métricas")
# Crea un cuadro de texto en la barra lateral para ingresar los símbolos de acciones 
simbolos_input = st.sidebar.text_input("Ingrese los símbolos de las acciones separados por comas (por ejemplo: AAPL,GOOGL,MSFT):", "AAPL,GOOGL,MSFT,AMZN,NVDA")
# Convierte el texto ingresado en una lista de símbolos
simbolos = [s.strip() for s in simbolos_input.split(',')]

# Selección del benchmark
benchmark_options = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "ACWI": "ACWI"
}
selected_benchmark = st.sidebar.selectbox("Seleccione el benchmark:", list(benchmark_options.keys()))
benchmark = benchmark_options[selected_benchmark]


# Para las pestañas

# Pestaña 1
with tab1:
    st.header("Análisis del activo")
    selected_asset = st.selectbox(
        "Seleccione un activo para analizar:", 
        simbolos, 
        key="selected_asset"  # Almacenamos en session_state
    )
    
    # Obtener datos y calcular rendimientos solo para el activo seleccionado
    df_precios = obtener_datos([selected_asset])  # Se pasa la lista con un solo activo
    df_rendimientos = calcular_rendimientos(df_precios)
    df_rendimientos_log = calcular_rendimientos_log(df_precios)

    # Diccionario para almacenar los promedios de rendimiento
    promedios_rendi_diario = {stock: df_rendimientos[stock].mean() for stock in [selected_asset]}
    
    # Calcular el sesgo y la kurtosis para el activo seleccionado
    skew_rendi_diario = df_rendimientos[selected_asset].skew()  # Sesgo para el activo seleccionado
    kurtosis_rendi_diario = df_rendimientos[selected_asset].kurtosis()  # Kurtosis para el activo seleccionado
    
    # Crear columnas para mostrar métricas
    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)
    # Mostrar métricas para el activo seleccionado
    if selected_asset:
        promedio_diario = promedios_rendi_diario[selected_asset]
        promedio_anualizado = (1 + promedio_diario) ** 252 - 1  # Convertir a rendimiento anualizado
        col1.metric("Rendimiento promedio diario", f"{promedio_diario:.5%}")
        col2.metric("Rendimiento anualizado", f"{promedio_anualizado:.2%}")
        col3.metric("Último precio en moneda de la acción correspondiente", f"${df_precios[selected_asset].iloc[-1]:.2f}")
        col4.metric(f"Sesgo de {selected_asset}", f"{skew_rendi_diario:.5f}")
        col5.metric(f"Kurtosis de {selected_asset}", f"{kurtosis_rendi_diario:.5f}")
    else:
        col1.metric("Rendimiento promedio diario", "N/A")
        col2.metric("Rendimiento anualizado", "N/A")
        col3.metric("Último precio", "N/A")
        col4.metric(f"Sesgo de {selected_asset}", "N/A")
        col5.metric(f"Kurtosis de {selected_asset}", "N/A")
    # Datos de países de inversión para ETFs

    etf_country_data = {
        'S&P 500': ['United States'],
        'Nasdaq': ['United States'],
        'Dow Jones': ['United States'],
        'Russell 2000': ['United States'],
        'ACWI': ['Global']
    }
    
    # Mostrar el ETF seleccionado en un metric
    #st.subheader("ETF seleccionado", selected_benchmark)


    all_symbols = simbolos + [benchmark]
    df_stocks = obtener_datos(all_symbols)
    returns, cumulative_returns, normalized_prices = calcular_metricas(df_stocks)
    
    fig_asset = go.Figure()
    fig_asset.add_trace(go.Scatter(x=normalized_prices.index, y=normalized_prices[selected_asset], name=selected_asset))
    fig_asset.add_trace(go.Scatter(x=normalized_prices.index, y=normalized_prices[benchmark], name=selected_benchmark))
    fig_asset.update_layout(title=f'Precio Normalizado: {selected_asset} vs {selected_benchmark} (Base 100)', xaxis_title='Fecha', yaxis_title='Precio Normalizado')
    st.plotly_chart(fig_asset, use_container_width=True, key="price_normalized")
        
    # Mostrar los datos en la página para el activo seleccionado
    st.subheader("Últimos 5 Datos de Precios")
    st.dataframe(df_precios.tail(5))
    
    st.subheader("Últimos 5 Rendimientos")
    st.dataframe(df_rendimientos.tail(5))
    
    st.subheader("Últimos 5 Rendimientos Logarítmicos")
    st.dataframe(df_rendimientos_log.tail(5))

# Pestaña 2
with tab2:
    with st.sidebar:
        st.header("Cálculo de VaR y CVaR")
        selected_asset = st.selectbox(
            "Seleccione el activo (debe ser el mismo que en Análisis del activo)", 
            simbolos, 
            index=simbolos.index(st.session_state["selected_asset"]),
            key="selected_asset_2"
        )
    # Mostrar sobre qué activo se está realizando el cálculo
    st.subheader(f"Análisis de VaR y CVaR para el activo: {selected_asset}")  
    # Selector de nivel de confianza
    alpha_options = {
        "95%": 0.95,
        "97.5%": 0.975,
        "99%": 0.99
    }
    selected_alpha_label = st.selectbox("Seleccione un nivel de confianza:", list(alpha_options.keys()))
    alpha = alpha_options[selected_alpha_label]
    
    # El percentil para el cálculo del VaR
    percentil = 100 - alpha * 100
    if selected_asset:
        mean = np.mean(df_rendimientos[selected_asset])
        stdev = np.std(df_rendimientos[selected_asset])
        
        # Paramétrico (Normal) VaR
        VaR_param = norm.ppf(1 - alpha, mean, stdev)
        
        # Historical VaR
        hVaR = df_rendimientos[selected_asset].quantile(1 - alpha)
        
        # Monte Carlo VaR
        n_sims = 100000
        np.random.seed(42)  # Para reproducibilidad
        sim_returns = np.random.normal(mean, stdev, n_sims)
        MCVaR = np.percentile(sim_returns, percentil)
        
        # CVaR (Expected Shortfall)
        CVaR = df_rendimientos[selected_asset][df_rendimientos[selected_asset] <= hVaR].mean()
        
        # Mostrar métricas en Streamlit
        st.subheader("Métricas de riesgo")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"{int(alpha*100)}% VaR (Paramétrico)", f"{VaR_param:.4%}")
        col2.metric(f"{int(alpha*100)}% VaR (Histórico)", f"{hVaR:.4%}")
        col3.metric(f"{int(alpha*100)}% VaR (Monte Carlo)", f"{MCVaR:.4%}")
        col4.metric(f"{int(alpha*100)}% CVaR", f"{CVaR:.4%}")
        
        # Visualización gráfica
        st.subheader("Gráfica métricas de riesgo")
        
        # Crear la figura y el eje
        fig, ax = plt.subplots(figsize=(13, 5))
        
        # Generar histograma
        n, bins, patches = ax.hist(df_rendimientos[selected_asset], bins=50, color='blue', alpha=0.7, label='Returns')
        
        # Identificar y colorear de rojo las barras a la izquierda de hVaR
        for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
            if bin_left < hVaR:
                patch.set_facecolor('red')
        
        # Marcar las líneas de VaR y CVaR
        ax.axvline(x=VaR_param, color='skyblue', linestyle='--', label=f'VaR {int(alpha*100)}% (Paramétrico)')
        ax.axvline(x=MCVaR, color='grey', linestyle='--', label=f'VaR {int(alpha*100)}% (Monte Carlo)')
        ax.axvline(x=hVaR, color='green', linestyle='--', label=f'VaR {int(alpha*100)}% (Histórico)')
        ax.axvline(x=CVaR, color='purple', linestyle='-.', label=f'CVaR {int(alpha*100)}%')
        
        # Configurar etiquetas y leyenda
        ax.set_title(f"Histograma de Rendimientos con VaR y CVaR para {selected_asset}")
        ax.set_xlabel("Rendimiento Diario")
        ax.set_ylabel("Frecuencia")
        ax.legend()
        
        # Mostrar la figura en Streamlit
        st.pyplot(fig)
        
        # Agregar explicación básica
        with st.expander("¿Qué significan estas métricas?"):
            st.write(f"""
            - **VaR {int(alpha*100)}%**: Con un nivel de confianza del {int(alpha*100)}%, se espera que la pérdida máxima diaria no exceda este valor.
            - **CVaR {int(alpha*100)}%**: Si se excede el VaR, esta es la pérdida promedio esperada.
            - **Métodos**:
                - Paramétrico: Asume distribución normal
                - Histórico: Basado en datos históricos reales
                - Monte Carlo: Simulación de {n_sims:,} escenarios
            """)

    else:
        st.write("Seleccione un activo para visualizar sus métricas de riesgo.")
        st.write("Seleccione un activo para visualizar sus métricas de riesgo.")

# Pestaña3

# Funciones para calcular VaR y CVaR
def calcular_var_historico(rendimientos, alpha):
    """VaR por método histórico: percentil de los rendimientos"""
    return np.percentile(rendimientos, (1 - alpha) * 100)

def calcular_cvar_historico(rendimientos, alpha):
    """CVaR por método histórico: media de pérdidas más extremas"""
    var = calcular_var_historico(rendimientos, alpha)
    return rendimientos[rendimientos <= var].mean()

def calcular_var_parametrico(rendimientos, alpha):
    """VaR paramétrico: usa la media y desviación estándar de los rendimientos"""
    mu, sigma = np.mean(rendimientos), np.std(rendimientos)
    z_score = stats.norm.ppf(1 - alpha)  # Cuantil de la normal
    return mu + z_score * sigma

def calcular_cvar_parametrico(rendimientos, alpha):
    """CVaR paramétrico: basado en la distribución normal"""
    mu, sigma = np.mean(rendimientos), np.std(rendimientos)
    z_score = stats.norm.ppf(1 - alpha)
    pdf_z = stats.norm.pdf(z_score)
    cvar = mu - sigma * (pdf_z / (1 - alpha))  # Fórmula de CVaR en normal
    return cvar

with tab3:
    st.header(f"VaR y CVaR para el activo: {selected_asset} con Rolling Windows (Histórico y Paramétrico)")

    # Nivel de confianza
    alpha_95 = 0.95
    alpha_99 = 0.99

    # Selección del tamaño de ventana
    window_size = st.slider("Seleccione tamaño de la ventana", 1, 252, 252)

    # Cálculo con método HISTÓRICO
    df_var_hist_95 = df_rendimientos.rolling(window=window_size).apply(lambda x: calcular_var_historico(x, alpha_95), raw=True)
    df_var_hist_99 = df_rendimientos.rolling(window=window_size).apply(lambda x: calcular_var_historico(x, alpha_99), raw=True)
    df_cvar_hist_95 = df_rendimientos.rolling(window=window_size).apply(lambda x: calcular_cvar_historico(x, alpha_95), raw=True)
    df_cvar_hist_99 = df_rendimientos.rolling(window=window_size).apply(lambda x: calcular_cvar_historico(x, alpha_99), raw=True)

    # Cálculo con método PARAMÉTRICO
    df_var_param_95 = df_rendimientos.rolling(window=window_size).apply(lambda x: calcular_var_parametrico(x, alpha_95), raw=True)
    df_var_param_99 = df_rendimientos.rolling(window=window_size).apply(lambda x: calcular_var_parametrico(x, alpha_99), raw=True)
    df_cvar_param_95 = df_rendimientos.rolling(window=window_size).apply(lambda x: calcular_cvar_parametrico(x, alpha_95), raw=True)
    df_cvar_param_99 = df_rendimientos.rolling(window=window_size).apply(lambda x: calcular_cvar_parametrico(x, alpha_99), raw=True)
    # Cálculo con método PARAMÉTRICO
    df_var_param_95 = df_rendimientos.rolling(window=window_size).apply(lambda x: calcular_var_parametrico(x, alpha), raw=True)
    df_var_param_99 = df_rendimientos.rolling(window=window_size).apply(lambda x: calcular_var_parametrico(x, alpha), raw=True)
    df_cvar_param_95 = df_rendimientos.rolling(window=window_size).apply(lambda x: calcular_cvar_parametrico(x, alpha), raw=True)
    df_cvar_param_99 = df_rendimientos.rolling(window=window_size).apply(lambda x: calcular_cvar_parametrico(x, alpha), raw=True)

    # Gráfico para método HISTÓRICO
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_var_hist_95, label="VaR 95% (Histórico)", color='red', linestyle='--')
    ax.plot(df_var_hist_99, label="VaR 99% (Histórico)", color='blue', linestyle='--')
    ax.plot(df_cvar_hist_95, label="CVaR 95% (Histórico)", color='purple', linestyle='-')
    ax.plot(df_cvar_hist_99, label="CVaR 99% (Histórico)", color='orange', linestyle='-')
    
    ax.set_title(f"VaR y CVaR de:  {selected_asset} - Método Histórico")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor en Riesgo")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Gráfico para método PARAMÉTRICO
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_var_param_95, label="VaR 95% (Paramétrico)", color='red', linestyle='--')
    ax.plot(df_var_param_99, label="VaR 99% (Paramétrico)", color='blue', linestyle='--')
    ax.plot(df_cvar_param_95, label="CVaR 95% (Paramétrico)", color='purple', linestyle='-')
    ax.plot(df_cvar_param_99, label="CVaR 99% (Paramétrico)", color='orange', linestyle='-')

    ax.set_title(f"VaR y CVaR de:  {selected_asset}- Método Paramétrico")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor en Riesgo")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


# Pestaña 4
with tab4:
    st.header(f"Evaluación de eficiencia de VaR para el activo:  {selected_asset} ")
    alpha = st.selectbox("Seleccione nivel de confianza", [0.95, 0.975, 0.99], index=0)
    #selected_asset = st.selectbox("Seleccione un activo", simbolos)
    df_rendimientos = calcular_rendimientos(df_precios[selected_asset])

    VaR = calcular_var(df_rendimientos, alpha)
    violaciones = df_rendimientos[df_rendimientos < VaR]
    porcentaje_violaciones = len(violaciones) / len(df_rendimientos) * 100

    st.metric("VaR estimado", f"{VaR:.4%}")
    st.metric("Número de violaciones", f"{len(violaciones)}")
    st.metric("Porcentaje de violaciones", f"{porcentaje_violaciones:.2f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
#    ax.plot(df_rendimientos.index, df_rendimientos["columna_rendimientos"], label="Rendimientos", color='blue')
    ax.axhline(y=VaR, color='red', linestyle='--', label=f'VaR {int(alpha*100)}%')
    ax.scatter(violaciones.index, violaciones, color='red', label='Violaciones', zorder=3)
    ax.legend()
    st.pyplot(fig)


# Pestaña 5
with tab5:
    st.header(f"VaR del activo:  {selected_asset} con Distribución Normal")

    # Definir niveles de significancia
    alpha_95 = 0.05
    alpha_99 = 0.01

    # Calcular los percentiles de la distribución normal
    q_95 = stats.norm.ppf(alpha_95)  # Percentil para 95%
    q_99 = stats.norm.ppf(alpha_99)  # Percentil para 99%

    # Tamaño de la ventana fija en 252 días
    window_size = st.slider("Seleccione tamaño de la ventana", 1, 252, 251)

    # Cálculo de la desviación estándar en ventana móvil de 252 días
    rolling_std = df_rendimientos.rolling(window=window_size).std()

    # Cálculo del VaR bajo distribución normal
    df_rolling_var_95 = q_95 * rolling_std
    df_rolling_var_99 = q_99 * rolling_std

    # Graficar los resultados
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_rolling_var_95, label="VaR 95% (Normal)", color='red')
    ax.plot(df_rolling_var_99, label="VaR 99% (Normal)", color='blue')
    ax.legend()
    
    ax.set_title(f"VaR del activo:  {selected_asset} asumiendo Distribución Normal")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor en Riesgo")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
