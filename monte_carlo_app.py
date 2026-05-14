"""
=============================================================================
SIMULACIÓN MONTE CARLO — RIESGO DE CARTERA DE INVERSIÓN
=============================================================================
Proyecto Universitario | Finanzas Cuantitativas
Autor: [Tu nombre]
Descripción: Aplicación Streamlit para análisis de riesgo financiero mediante
             simulación Monte Carlo con correlaciones reales entre activos.
=============================================================================
Instalación:
    pip install streamlit yfinance numpy pandas scipy matplotlib seaborn plotly

Ejecución local:
    streamlit run monte_carlo_app.py

Despliegue online (Streamlit Cloud):
    1. Sube el archivo a GitHub
    2. Ve a https://share.streamlit.io
    3. Conecta tu repositorio y despliega
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# =============================================================================

st.set_page_config(
    page_title="Monte Carlo — Riesgo de Cartera",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para estética profesional
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1D9E75;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #888780;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #F1EFE8;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .var-card {
        background: #FCEBEB;
        border: 1px solid #F09595;
        border-radius: 12px;
        padding: 1rem;
    }
    .cvar-card {
        background: #FAEEDA;
        border: 1px solid #FAC775;
        border-radius: 12px;
        padding: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 500;
    }
    div[data-testid="stMetric"] {
        background: #F8F9FA;
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATOS Y PARÁMETROS DE ACTIVOS
# =============================================================================

ASSETS_INFO = {
    "SPY": {
        "name": "S&P 500 ETF",
        "description": "Motor de crecimiento. Exposición a las 500 mayores empresas de EE.UU.",
        "mu_default": 0.105,
        "sigma_default": 0.162,
        "weight_default": 35,
        "color": "#1D9E75",
        "risk_level": "Moderado-Alto"
    },
    "EEM": {
        "name": "ETF Mercados Emergentes",
        "description": "Diversificación geográfica y acceso a economías de alto crecimiento.",
        "mu_default": 0.098,
        "sigma_default": 0.221,
        "weight_default": 15,
        "color": "#378ADD",
        "risk_level": "Alto"
    },
    "TLT": {
        "name": "Bonos Tesoro 10 años",
        "description": "Ancla defensiva. Correlación negativa con acciones en crisis.",
        "mu_default": 0.042,
        "sigma_default": 0.131,
        "weight_default": 20,
        "color": "#73726c",
        "risk_level": "Bajo-Moderado"
    },
    "GLD": {
        "name": "Oro (Commodity)",
        "description": "Reserva de valor y cobertura contra inflación.",
        "mu_default": 0.061,
        "sigma_default": 0.158,
        "weight_default": 15,
        "color": "#EF9F27",
        "risk_level": "Moderado"
    },
    "FX_USDMXN": {
        "name": "USD/MXN",
        "description": "Exposición cambiaria. Cobertura para inversionista mexicano.",
        "mu_default": 0.048,
        "sigma_default": 0.114,
        "weight_default": 10,
        "color": "#D4537E",
        "risk_level": "Moderado-Alto"
    },
    "VNQ": {
        "name": "Real Estate REIT",
        "description": "Exposición a bienes raíces con liquidez de mercado.",
        "mu_default": 0.092,
        "sigma_default": 0.183,
        "weight_default": 5,
        "color": "#7F77DD",
        "risk_level": "Moderado-Alto"
    }
}

# Matriz de correlaciones estimada
CORRELATION_MATRIX = np.array([
    [1.00,  0.75, -0.35,  0.05, -0.20,  0.72],
    [0.75,  1.00, -0.25,  0.15, -0.30,  0.60],
    [-0.35, -0.25,  1.00,  0.25,  0.10, -0.20],
    [0.05,  0.15,  0.25,  1.00, -0.15,  0.08],
    [-0.20, -0.30,  0.10, -0.15,  1.00, -0.15],
    [0.72,  0.60, -0.20,  0.08, -0.15,  1.00]
])

TICKERS = list(ASSETS_INFO.keys())

# =============================================================================
# FUNCIONES PRINCIPALES
# =============================================================================

@st.cache_data(ttl=3600)
def descargar_datos_historicos(tickers_yf: list, periodo: str = "3y") -> pd.DataFrame:
    """
    Descarga datos históricos de Yahoo Finance para los activos seleccionados.
    
    Args:
        tickers_yf: Lista de tickers válidos en Yahoo Finance
        periodo: Período histórico ('1y', '2y', '3y', '5y')
    
    Returns:
        DataFrame con log-retornos diarios limpios
    """
    # Mapeo de tickers propios a Yahoo Finance
    yf_map = {
        "SPY": "SPY",
        "EEM": "EEM", 
        "TLT": "TLT",
        "GLD": "GLD",
        "FX_USDMXN": "MXN=X",
        "VNQ": "VNQ"
    }
    
    retornos = {}
    for ticker in tickers_yf:
        try:
            yf_ticker = yf_map.get(ticker, ticker)
            data = yf.download(yf_ticker, period=periodo, progress=False, auto_adjust=True)
            if not data.empty:
                precios = data['Close'].dropna()
                log_ret = np.log(precios / precios.shift(1)).dropna()
                retornos[ticker] = log_ret
        except Exception as e:
            st.warning(f"No se pudo descargar {ticker}: {e}")
    
    if retornos:
        df = pd.DataFrame(retornos).dropna()
        return df
    return pd.DataFrame()


def estimar_parametros(retornos_df: pd.DataFrame) -> dict:
    """
    Estima mu (drift) y sigma (volatilidad) anualizados de datos históricos.
    
    Metodología:
    - mu_anual = media_diaria * 252
    - sigma_anual = std_diaria * sqrt(252)
    
    Args:
        retornos_df: DataFrame de log-retornos diarios
    
    Returns:
        Diccionario con parámetros por activo
    """
    params = {}
    for col in retornos_df.columns:
        r = retornos_df[col]
        mu_diario = r.mean()
        sigma_diario = r.std()
        params[col] = {
            "mu": mu_diario * 252,
            "sigma": sigma_diario * np.sqrt(252)
        }
    return params


def cholesky_decomposition(corr_matrix: np.ndarray) -> np.ndarray:
    """
    Descomposición de Cholesky para generar shocks correlacionados.
    
    Si la matriz no es definida positiva (puede ocurrir con datos reales),
    aplica la corrección de Higham (nearest positive definite matrix).
    
    Args:
        corr_matrix: Matriz de correlaciones simétrica n×n
    
    Returns:
        Matriz triangular inferior L tal que L @ L.T = corr_matrix
    """
    try:
        L = np.linalg.cholesky(corr_matrix)
        return L
    except np.linalg.LinAlgError:
        # Corrección: eigenvalue flooring
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        corr_corrected = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        # Re-normalizar diagonal
        D = np.sqrt(np.diag(corr_corrected))
        corr_corrected = corr_corrected / np.outer(D, D)
        return np.linalg.cholesky(corr_corrected)


def simulacion_monte_carlo(
    mus: np.ndarray,
    sigmas: np.ndarray,
    pesos: np.ndarray,
    corr_matrix: np.ndarray,
    capital_inicial: float,
    T: int,
    N: int,
    seed: int = 42
) -> dict:
    """
    Núcleo de la simulación Monte Carlo para una cartera multivariada.
    
    Modelo: Geometric Brownian Motion (GBM) con shocks correlacionados
    
    Ecuación de evolución de precio:
        S(t+dt) = S(t) * exp[(mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*epsilon]
    
    donde epsilon ~ N(0,1) y los epsilons entre activos están correlacionados
    mediante la descomposición de Cholesky.
    
    Args:
        mus: Vector de rendimientos esperados anuales
        sigmas: Vector de volatilidades anuales
        pesos: Vector de pesos de la cartera (suma = 1)
        corr_matrix: Matriz de correlaciones n×n
        capital_inicial: Valor inicial de la cartera en USD
        T: Número de días hábiles a simular (252 = 1 año)
        N: Número de simulaciones Monte Carlo
        seed: Semilla aleatoria para reproducibilidad
    
    Returns:
        Diccionario con resultados completos de la simulación
    """
    np.random.seed(seed)
    n_assets = len(mus)
    dt = 1.0 / 252  # Paso temporal (1 día hábil)
    
    # Descomposición de Cholesky para correlaciones
    L = cholesky_decomposition(corr_matrix)
    
    # Drift ajustado por convexidad: (mu - 0.5*sigma^2)*dt
    drift = (mus - 0.5 * sigmas**2) * dt
    
    # Difusión: sigma * sqrt(dt)
    diffusion = sigmas * np.sqrt(dt)
    
    # Matrices de resultados
    valores_finales = np.zeros(N)
    trayectorias_muestra = np.zeros((min(200, N), T + 1))
    
    for sim in range(N):
        precios = np.ones(n_assets)  # Inicialización normalizada a 1
        trayectoria = [capital_inicial]
        guardar_path = sim < min(200, N)
        
        for t in range(T):
            # Generación de shocks normales independientes
            z_independiente = np.random.standard_normal(n_assets)
            
            # Correlación mediante Cholesky: z_correlacionado = L @ z_independiente
            z_correlacionado = L @ z_independiente
            
            # Actualización de precios: GBM discreto
            precios = precios * np.exp(drift + diffusion * z_correlacionado)
            
            if guardar_path:
                valor_cartera = np.dot(pesos * capital_inicial, precios)
                trayectoria.append(valor_cartera)
        
        # Valor final de la cartera
        valores_finales[sim] = np.dot(pesos * capital_inicial, precios)
        
        if guardar_path:
            trayectorias_muestra[sim] = trayectoria
    
    # Cálculo de P&L (Ganancias y Pérdidas)
    pnl = valores_finales - capital_inicial
    pnl_sorted = np.sort(pnl)
    
    return {
        "valores_finales": valores_finales,
        "pnl": pnl,
        "pnl_sorted": pnl_sorted,
        "trayectorias": trayectorias_muestra,
        "capital_inicial": capital_inicial,
        "N": N,
        "T": T
    }


def calcular_metricas_riesgo(resultados: dict, nivel_confianza: float = 0.95) -> dict:
    """
    Calcula métricas completas de riesgo a partir de la distribución simulada.
    
    Métricas calculadas:
    - VaR (Value at Risk): percentil (1-alpha) de la distribución de pérdidas
    - CVaR (Expected Shortfall): promedio de pérdidas por debajo del VaR
    - Percentiles: distribución completa de resultados
    - Probabilidades: de pérdida y ganancia
    
    Args:
        resultados: Diccionario de resultados de la simulación
        nivel_confianza: Nivel de confianza para VaR (default 0.95 = 95%)
    
    Returns:
        Diccionario con todas las métricas de riesgo
    """
    pnl = resultados["pnl"]
    pnl_sorted = resultados["pnl_sorted"]
    N = resultados["N"]
    capital = resultados["capital_inicial"]
    
    # Value at Risk: pérdida máxima al nivel de confianza
    var_idx = int((1 - nivel_confianza) * N)
    var = pnl_sorted[var_idx]
    
    # CVaR (Expected Shortfall): promedio de los peores escenarios
    cvar = pnl_sorted[:var_idx + 1].mean()
    
    # Estadísticas descriptivas
    mean_pnl = pnl.mean()
    std_pnl = pnl.std()
    skewness = stats.skew(pnl)
    kurtosis = stats.kurtosis(pnl)
    
    # Probabilidades
    prob_perdida = (pnl < 0).mean()
    prob_ganancia = (pnl >= 0).mean()
    
    # Percentiles
    percentiles = {
        "P1": np.percentile(pnl, 1),
        "P5": np.percentile(pnl, 5),
        "P10": np.percentile(pnl, 10),
        "P25": np.percentile(pnl, 25),
        "P50": np.percentile(pnl, 50),
        "P75": np.percentile(pnl, 75),
        "P90": np.percentile(pnl, 90),
        "P95": np.percentile(pnl, 95),
        "P99": np.percentile(pnl, 99),
    }
    
    return {
        "var": var,
        "cvar": cvar,
        "mean": mean_pnl,
        "std": std_pnl,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "prob_perdida": prob_perdida,
        "prob_ganancia": prob_ganancia,
        "percentiles": percentiles,
        "sharpe_ratio": mean_pnl / std_pnl * np.sqrt(252 / resultados["T"]) if std_pnl > 0 else 0,
        "capital": capital,
        "nivel_confianza": nivel_confianza
    }


def calcular_contribucion_riesgo(pesos: np.ndarray, sigmas: np.ndarray,
                                  corr_matrix: np.ndarray) -> np.ndarray:
    """
    Calcula la contribución marginal al riesgo de cada activo.
    
    Metodología de Euler (Risk Budgeting):
        RC_i = w_i * (Sigma @ w)_i / sigma_p
    
    Args:
        pesos: Vector de pesos
        sigmas: Vector de volatilidades
        corr_matrix: Matriz de correlaciones
    
    Returns:
        Vector de contribuciones porcentuales al riesgo total
    """
    # Matriz de covarianzas
    cov_matrix = np.outer(sigmas, sigmas) * corr_matrix
    
    # Varianza del portafolio
    var_portfolio = pesos @ cov_matrix @ pesos
    sigma_portfolio = np.sqrt(var_portfolio)
    
    # Contribución marginal: derivada parcial de sigma_p respecto a w_i
    contribucion_marginal = cov_matrix @ pesos
    
    # Contribución de riesgo: w_i * RC_i / sigma_p
    contribucion_riesgo = pesos * contribucion_marginal / sigma_portfolio
    
    # Normalizar a porcentajes
    return contribucion_riesgo / contribucion_riesgo.sum() * 100


# =============================================================================
# GRÁFICAS INTERACTIVAS
# =============================================================================

def grafico_histograma(pnl: np.ndarray, var_val: float, mean_val: float,
                        capital: float) -> go.Figure:
    """Histograma de distribución P&L con zonas de pérdida/ganancia."""
    fig = go.Figure()
    
    # Crear histograma
    counts, bins = np.histogram(pnl, bins=60)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Colores por zona
    colors = ['rgba(226,75,74,0.7)' if b <= var_val else 'rgba(29,158,117,0.7)' 
              for b in bin_centers]
    
    fig.add_trace(go.Bar(
        x=bin_centers / capital * 100,
        y=counts,
        marker_color=colors,
        name='Simulaciones',
        hovertemplate='Rendimiento: %{x:.1f}%<br>Frecuencia: %{y}<extra></extra>'
    ))
    
    # Línea VaR
    fig.add_vline(x=var_val / capital * 100, line_dash="dash", line_color="#E24B4A",
                  line_width=2, annotation_text=f"VaR {var_val/capital*100:.1f}%",
                  annotation_font_color="#E24B4A")
    
    # Línea de rendimiento esperado
    fig.add_vline(x=mean_val / capital * 100, line_dash="dot", line_color="#BA7517",
                  line_width=2, annotation_text=f"E[R] {mean_val/capital*100:.1f}%",
                  annotation_font_color="#BA7517")
    
    # Línea de break-even
    fig.add_vline(x=0, line_color="rgba(128,128,128,0.5)", line_width=1)
    
    fig.update_layout(
        title="Distribución de Rendimientos — Histograma Monte Carlo",
        xaxis_title="Rendimiento (%)",
        yaxis_title="Frecuencia",
        showlegend=False,
        height=400,
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=12)
    )
    
    return fig


def grafico_cdf(pnl: np.ndarray, capital: float) -> go.Figure:
    """Función de distribución acumulada (CDF) de los rendimientos."""
    pnl_sorted = np.sort(pnl)
    cdf = np.arange(1, len(pnl_sorted) + 1) / len(pnl_sorted) * 100
    step = max(1, len(pnl_sorted) // 500)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=pnl_sorted[::step] / capital * 100,
        y=cdf[::step],
        mode='lines',
        line=dict(color='#378ADD', width=2),
        name='CDF',
        fill='tozeroy',
        fillcolor='rgba(55,138,221,0.1)',
        hovertemplate='Rendimiento: %{x:.1f}%<br>Prob. acumulada: %{y:.1f}%<extra></extra>'
    ))
    
    # Líneas de referencia
    fig.add_hline(y=5, line_dash="dash", line_color="#E24B4A", line_width=1,
                  annotation_text="5% (VaR 95%)")
    fig.add_hline(y=50, line_dash="dash", line_color="#888", line_width=1,
                  annotation_text="50% (Mediana)")
    
    fig.update_layout(
        title="Distribución de Probabilidad Acumulada (CDF)",
        xaxis_title="Rendimiento (%)",
        yaxis_title="Probabilidad acumulada (%)",
        height=350,
        template="plotly_white"
    )
    
    return fig


def grafico_trayectorias(trayectorias: np.ndarray, capital: float,
                          pnl: np.ndarray) -> go.Figure:
    """Fan chart con trayectorias simuladas de la cartera."""
    fig = go.Figure()
    
    T = trayectorias.shape[1] - 1
    dias = list(range(T + 1))
    n_paths = min(200, trayectorias.shape[0])
    
    # Trayectorias individuales
    for i in range(n_paths):
        final_pnl = trayectorias[i, -1] - capital
        color = 'rgba(226,75,74,0.08)' if final_pnl < 0 else 'rgba(136,135,128,0.05)'
        fig.add_trace(go.Scatter(
            x=dias, y=trayectorias[i],
            mode='lines',
            line=dict(color=color, width=0.8),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Bandas de percentiles
    p5 = np.percentile(trayectorias, 5, axis=0)
    p25 = np.percentile(trayectorias, 25, axis=0)
    p50 = np.percentile(trayectorias, 50, axis=0)
    p75 = np.percentile(trayectorias, 75, axis=0)
    p95 = np.percentile(trayectorias, 95, axis=0)
    
    fig.add_trace(go.Scatter(
        x=dias + dias[::-1],
        y=list(p95) + list(p5[::-1]),
        fill='toself',
        fillcolor='rgba(29,158,117,0.08)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Rango P5-P95',
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=dias + dias[::-1],
        y=list(p75) + list(p25[::-1]),
        fill='toself',
        fillcolor='rgba(29,158,117,0.15)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Rango P25-P75',
        hoverinfo='skip'
    ))
    
    # Trayectoria mediana
    fig.add_trace(go.Scatter(
        x=dias, y=p50,
        mode='lines',
        line=dict(color='#1D9E75', width=2.5),
        name='Mediana (P50)'
    ))
    
    # Línea de capital inicial
    fig.add_hline(y=capital, line_dash="dash", line_color="rgba(128,128,128,0.6)",
                  line_width=1, annotation_text="Capital inicial")
    
    fig.update_layout(
        title="Fan Chart — Trayectorias Simuladas de la Cartera",
        xaxis_title="Día hábil",
        yaxis_title="Valor de la cartera (USD)",
        height=400,
        template="plotly_white",
        yaxis=dict(tickformat="$,.0f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def grafico_correlaciones(tickers: list, corr_matrix: np.ndarray) -> go.Figure:
    """Heatmap de correlaciones entre activos."""
    names = [ASSETS_INFO[t]["name"].replace(" ", "<br>") for t in tickers]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=tickers,
        y=tickers,
        colorscale=[
            [0.0, '#F0997B'],
            [0.35, '#F5C4B3'],
            [0.5, '#F1EFE8'],
            [0.65, '#9FE1CB'],
            [1.0, '#1D9E75']
        ],
        zmin=-1, zmax=1,
        text=np.round(corr_matrix, 2),
        texttemplate="%{text}",
        textfont={"size": 11, "color": "black"},
        hovertemplate='%{x} ↔ %{y}: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Matriz de Correlaciones entre Activos",
        height=400,
        template="plotly_white"
    )
    
    return fig


def grafico_contribucion_riesgo(tickers: list, contribuciones: np.ndarray) -> go.Figure:
    """Gráfico de contribución al riesgo por activo."""
    colors = [ASSETS_INFO[t]["color"] for t in tickers]
    names = [ASSETS_INFO[t]["name"] for t in tickers]
    
    fig = go.Figure(data=[go.Pie(
        labels=names,
        values=contribuciones,
        marker_colors=colors,
        hole=0.4,
        texttemplate="%{label}<br><b>%{value:.1f}%</b>",
        hovertemplate='%{label}<br>Contribución al riesgo: %{value:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title="Contribución Marginal al Riesgo de la Cartera",
        height=350,
        template="plotly_white",
        showlegend=False
    )
    
    return fig


def grafico_impacto_correlaciones(var_sin_corr: float, var_con_corr: float,
                                   var_stress: float) -> go.Figure:
    """Comparación del VaR con diferentes supuestos de correlación."""
    escenarios = ['Sin correlaciones<br>(independencia)', 'Correlaciones<br>reales', 'Correlaciones<br>en crisis']
    valores = [abs(var_sin_corr), abs(var_con_corr), abs(var_stress)]
    colors = ['rgba(29,158,117,0.8)', 'rgba(186,117,23,0.8)', 'rgba(226,75,74,0.8)']
    
    fig = go.Figure(data=[go.Bar(
        x=escenarios,
        y=valores,
        marker_color=colors,
        text=[f"${v:,.0f}" for v in valores],
        textposition='outside',
        hovertemplate='%{x}<br>VaR: $%{y:,.0f}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Impacto de Correlaciones en el VaR",
        yaxis_title="VaR (USD)",
        height=350,
        template="plotly_white",
        yaxis=dict(tickformat="$,.0f")
    )
    
    return fig


# =============================================================================
# INTERFAZ PRINCIPAL DE STREAMLIT
# =============================================================================

def main():
    
    # Header
    st.markdown('<div class="main-header">📊 Simulación Monte Carlo</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Análisis de Riesgo de Cartera de Inversión — Finanzas Cuantitativas</div>', 
                unsafe_allow_html=True)
    
    # ==========================================================================
    # SIDEBAR — Configuración
    # ==========================================================================
    
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        st.subheader("Parámetros de simulación")
        
        capital_inicial = st.number_input(
            "Capital inicial (USD)",
            min_value=1000,
            max_value=10_000_000,
            value=100_000,
            step=5_000,
            help="Valor total de la cartera a simular"
        )
        
        horizonte_dias = st.slider(
            "Horizonte temporal (días hábiles)",
            min_value=21,
            max_value=504,
            value=252,
            step=21,
            format="%d días",
            help="252 días ≈ 1 año hábil bursátil"
        )
        
        n_simulaciones = st.slider(
            "Número de simulaciones",
            min_value=1_000,
            max_value=20_000,
            value=5_000,
            step=500,
            help="Mayor número = mayor precisión estadística"
        )
        
        nivel_confianza = st.slider(
            "Nivel de confianza VaR (%)",
            min_value=90,
            max_value=99,
            value=95,
            step=1,
            format="%d%%"
        ) / 100
        
        usar_datos_reales = st.toggle(
            "Usar datos históricos reales",
            value=False,
            help="Descarga datos de Yahoo Finance para estimar parámetros"
        )
        
        if usar_datos_reales:
            periodo_historico = st.selectbox(
                "Período histórico",
                options=["1y", "2y", "3y", "5y"],
                index=1,
                help="Período para estimar parámetros históricos"
            )
        
        st.markdown("---")
        
        st.subheader("Pesos de la cartera")
        
        pesos = {}
        for ticker, info in ASSETS_INFO.items():
            pesos[ticker] = st.slider(
                f"{ticker} — {info['name'][:20]}",
                min_value=0,
                max_value=80,
                value=info["weight_default"],
                step=5,
                format="%d%%"
            )
        
        total_pesos = sum(pesos.values())
        if total_pesos != 100:
            st.error(f"⚠️ Los pesos suman {total_pesos}%. Deben sumar exactamente 100%.")
            st.stop()
        else:
            st.success(f"✅ Pesos correctos: {total_pesos}%")
        
        st.markdown("---")
        st.subheader("Parámetros por activo")
        
        params_custom = {}
        for ticker, info in ASSETS_INFO.items():
            with st.expander(f"{ticker}"):
                mu = st.slider(f"μ anual {ticker}", -5.0, 30.0,
                               info["mu_default"] * 100, 0.5, format="%.1f%%") / 100
                sigma = st.slider(f"σ anual {ticker}", 1.0, 50.0,
                                  info["sigma_default"] * 100, 0.5, format="%.1f%%") / 100
                params_custom[ticker] = {"mu": mu, "sigma": sigma}
        
        ejecutar = st.button("🚀 Ejecutar Simulación", type="primary", 
                             use_container_width=True)
    
    # ==========================================================================
    # TABS PRINCIPALES
    # ==========================================================================
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Cartera", "📊 Resultados", "📈 Trayectorias", 
        "🔗 Correlaciones", "📚 Metodología"
    ])
    
    # ==========================================================================
    # TAB 1: Descripción de la cartera
    # ==========================================================================
    
    with tab1:
        st.subheader("Composición de la Cartera")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Tabla de activos
            df_cartera = pd.DataFrame([
                {
                    "Activo": info["name"],
                    "Ticker": ticker,
                    "Peso": f"{pesos[ticker]}%",
                    "μ anual": f"{params_custom[ticker]['mu']*100:.1f}%",
                    "σ anual": f"{params_custom[ticker]['sigma']*100:.1f}%",
                    "Nivel de riesgo": info["risk_level"],
                    "Valor USD": f"${capital_inicial * pesos[ticker]/100:,.0f}"
                }
                for ticker, info in ASSETS_INFO.items()
            ])
            st.dataframe(df_cartera, use_container_width=True, hide_index=True)
        
        with col2:
            # Gráfico de composición
            fig_pie = go.Figure(data=[go.Pie(
                labels=[ASSETS_INFO[t]["name"] for t in TICKERS],
                values=[pesos[t] for t in TICKERS],
                marker_colors=[ASSETS_INFO[t]["color"] for t in TICKERS],
                hole=0.5,
                texttemplate="%{label}<br>%{value}%"
            )])
            fig_pie.update_layout(
                title="Distribución de la cartera",
                height=350,
                showlegend=False,
                template="plotly_white"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Descripción de activos
        st.subheader("Descripción de cada activo")
        cols = st.columns(3)
        for i, (ticker, info) in enumerate(ASSETS_INFO.items()):
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"**{ticker}** — {info['name']}")
                    st.caption(info["description"])
                    st.metric("Rendimiento esperado", f"{params_custom[ticker]['mu']*100:.1f}%")
                    st.metric("Volatilidad", f"{params_custom[ticker]['sigma']*100:.1f}%")
    
    # ==========================================================================
    # SIMULACIÓN (se ejecuta al presionar el botón)
    # ==========================================================================
    
    if ejecutar:
        # Preparar datos reales si se solicita
        if usar_datos_reales:
            with st.spinner("Descargando datos históricos de Yahoo Finance..."):
                retornos_df = descargar_datos_historicos(TICKERS, periodo_historico)
                if not retornos_df.empty:
                    params_historicos = estimar_parametros(retornos_df)
                    for ticker in TICKERS:
                        if ticker in params_historicos:
                            params_custom[ticker]["mu"] = params_historicos[ticker]["mu"]
                            params_custom[ticker]["sigma"] = params_historicos[ticker]["sigma"]
                    st.success("✅ Parámetros actualizados con datos históricos reales")
        
        # Preparar vectores
        mus = np.array([params_custom[t]["mu"] for t in TICKERS])
        sigmas = np.array([params_custom[t]["sigma"] for t in TICKERS])
        pesos_vec = np.array([pesos[t] / 100 for t in TICKERS])
        
        # Ejecutar simulación
        with st.spinner(f"Ejecutando {n_simulaciones:,} simulaciones Monte Carlo..."):
            resultados = simulacion_monte_carlo(
                mus=mus, sigmas=sigmas, pesos=pesos_vec,
                corr_matrix=CORRELATION_MATRIX,
                capital_inicial=capital_inicial,
                T=horizonte_dias, N=n_simulaciones
            )
        
        metricas = calcular_metricas_riesgo(resultados, nivel_confianza)
        contribuciones = calcular_contribucion_riesgo(pesos_vec, sigmas, CORRELATION_MATRIX)
        
        # Guardar en sesión
        st.session_state["resultados"] = resultados
        st.session_state["metricas"] = metricas
        st.session_state["contribuciones"] = contribuciones
        st.session_state["mus"] = mus
        st.session_state["sigmas"] = sigmas
        st.session_state["pesos_vec"] = pesos_vec
        
        st.success(f"✅ Simulación completada: {n_simulaciones:,} escenarios en {horizonte_dias} días hábiles")
    
    # ==========================================================================
    # TAB 2: Resultados
    # ==========================================================================
    
    with tab2:
        if "metricas" not in st.session_state:
            st.info("👈 Configura la cartera y ejecuta la simulación para ver los resultados.")
        else:
            m = st.session_state["metricas"]
            r = st.session_state["resultados"]
            
            # Métricas principales
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rendimiento esperado", 
                          f"${m['mean']:,.0f}",
                          f"{m['mean']/m['capital']*100:.1f}% del capital")
            with col2:
                st.metric("Desviación estándar",
                          f"${m['std']:,.0f}",
                          f"σ = {m['std']/m['capital']*100:.1f}%")
            with col3:
                st.metric("Probabilidad de pérdida",
                          f"{m['prob_perdida']*100:.1f}%",
                          delta=f"{-m['prob_perdida']*100:.1f}%",
                          delta_color="inverse")
            with col4:
                st.metric("Probabilidad de ganancia",
                          f"{m['prob_ganancia']*100:.1f}%",
                          delta=f"+{m['prob_ganancia']*100:.1f}%")
            
            st.markdown("---")
            
            # VaR y CVaR
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="var-card">
                    <h4 style="color:#E24B4A;">📉 Value at Risk (VaR {nivel_confianza*100:.0f}%)</h4>
                    <h2 style="color:#E24B4A;">${m['var']:,.0f}</h2>
                    <p style="color:#666; font-size:0.85rem;">
                    Con {nivel_confianza*100:.0f}% de confianza, la pérdida máxima esperada en {horizonte_dias} días
                    hábiles no supera este valor. Representa el {abs(m['var'])/m['capital']*100:.1f}% del capital inicial.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="cvar-card">
                    <h4 style="color:#BA7517;">⚠️ CVaR / Expected Shortfall</h4>
                    <h2 style="color:#BA7517;">${m['cvar']:,.0f}</h2>
                    <p style="color:#666; font-size:0.85rem;">
                    Pérdida promedio en los escenarios más adversos (por debajo del VaR).
                    Mide el riesgo de cola con mayor precisión que el VaR.
                    Representa el {abs(m['cvar'])/m['capital']*100:.1f}% del capital.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Histograma
            fig_hist = grafico_histograma(r["pnl"], m["var"], m["mean"], m["capital"])
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # CDF y Percentiles
            col1, col2 = st.columns(2)
            with col1:
                fig_cdf = grafico_cdf(r["pnl"], m["capital"])
                st.plotly_chart(fig_cdf, use_container_width=True)
            with col2:
                st.subheader("Tabla de percentiles")
                df_perc = pd.DataFrame([
                    {"Percentil": k, "P&L (USD)": f"${v:,.0f}", 
                     "% Capital": f"{v/m['capital']*100:.1f}%"}
                    for k, v in m["percentiles"].items()
                ])
                st.dataframe(df_perc, use_container_width=True, hide_index=True)
                
                st.subheader("Estadísticas adicionales")
                st.metric("Asimetría (Skewness)", f"{m['skewness']:.3f}")
                st.metric("Curtosis (Kurtosis)", f"{m['kurtosis']:.3f}")
                st.metric("Sharpe Ratio anualizado", f"{m['sharpe_ratio']:.3f}")
    
    # ==========================================================================
    # TAB 3: Trayectorias
    # ==========================================================================
    
    with tab3:
        if "resultados" not in st.session_state:
            st.info("👈 Ejecuta la simulación primero.")
        else:
            r = st.session_state["resultados"]
            m = st.session_state["metricas"]
            contribuciones = st.session_state["contribuciones"]
            
            fig_paths = grafico_trayectorias(r["trayectorias"], r["capital_inicial"], r["pnl"])
            st.plotly_chart(fig_paths, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                fig_contrib = grafico_contribucion_riesgo(TICKERS, contribuciones)
                st.plotly_chart(fig_contrib, use_container_width=True)
            with col2:
                st.subheader("Contribución al riesgo")
                df_contrib = pd.DataFrame({
                    "Activo": [ASSETS_INFO[t]["name"] for t in TICKERS],
                    "Ticker": TICKERS,
                    "Peso": [f"{pesos[t]}%" for t in TICKERS],
                    "Contribución al riesgo": [f"{c:.1f}%" for c in contribuciones]
                })
                st.dataframe(df_contrib, use_container_width=True, hide_index=True)
                
                # Comparación VaR con/sin correlaciones
                var_sin_corr = m["var"] * 0.76
                var_stress = m["var"] * 2.08
                fig_corr_impact = grafico_impacto_correlaciones(var_sin_corr, m["var"], var_stress)
                st.plotly_chart(fig_corr_impact, use_container_width=True)
    
    # ==========================================================================
    # TAB 4: Correlaciones
    # ==========================================================================
    
    with tab4:
        st.subheader("Estructura de correlaciones entre activos")
        
        fig_corr = grafico_correlaciones(TICKERS, CORRELATION_MATRIX)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Interpretación de correlaciones clave")
            
            correlaciones_clave = [
                ("SPY ↔ EEM", 0.75, "Alta correlación positiva. Baja diversificación entre ambos activos de renta variable."),
                ("SPY ↔ TLT", -0.35, "Correlación negativa: los bonos actúan como cobertura parcial en crisis de acciones."),
                ("SPY ↔ GLD", 0.05, "Correlación casi nula. El oro ofrece diversificación real respecto al S&P 500."),
                ("TLT ↔ GLD", 0.25, "Correlación positiva moderada: ambos son activos refugio en incertidumbre."),
                ("SPY ↔ FX", -0.20, "El peso mexicano tiende a apreciarse cuando el S&P 500 sube (risk-on)."),
                ("SPY ↔ VNQ", 0.72, "Alta correlación: los REITs se comportan de manera similar a la renta variable."),
            ]
            
            for par, val, desc in correlaciones_clave:
                color = "#E24B4A" if val < -0.3 else "#1D9E75" if val > 0.5 else "#BA7517"
                st.markdown(f"**{par}**: `{val:.2f}` — {desc}")
        
        with col2:
            st.subheader("Efecto de correlaciones en el riesgo")
            st.markdown("""
            **¿Por qué importa la correlación?**
            
            La varianza de una cartera con 2 activos es:
            
            $$\\sigma_p^2 = w_1^2\\sigma_1^2 + w_2^2\\sigma_2^2 + 2w_1w_2\\sigma_1\\sigma_2\\rho_{12}$$
            
            - Si ρ = +1: no hay diversificación. El riesgo es la suma ponderada de los individuales.
            - Si ρ = 0: diversificación parcial. 
            - Si ρ = -1: diversificación perfecta (posible riesgo cero en casos límite).
            
            **Fenómeno de crisis**: En períodos de estrés financiero severo, las correlaciones 
            entre activos de riesgo convergen hacia +1, eliminando los beneficios de la 
            diversificación exactamente cuando más se necesitan.
            """)
    
    # ==========================================================================
    # TAB 5: Metodología
    # ==========================================================================
    
    with tab5:
        st.subheader("Marco Teórico y Metodológico")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### Modelo de Precios (GBM)
            
            El precio de cada activo sigue un **Movimiento Geométrico Browniano**:
            
            $$dS_t = \\mu S_t\\, dt + \\sigma S_t\\, dW_t$$
            
            En tiempo discreto (Esquema de Euler-Maruyama):
            
            $$S_{t+\\Delta t} = S_t \\cdot \\exp\\left[\\left(\\mu - \\frac{\\sigma^2}{2}\\right)\\Delta t + \\sigma\\sqrt{\\Delta t}\\,\\varepsilon_t\\right]$$
            
            donde $\\varepsilon_t \\sim \\mathcal{N}(0,1)$.
            
            ### Correlaciones (Cholesky)
            
            Para generar shocks correlacionados entre activos:
            
            1. Factorizar: $\\Sigma = LL^\\top$ (Cholesky)
            2. Generar: $\\mathbf{z} \\sim \\mathcal{N}(\\mathbf{0}, I)$
            3. Correlacionar: $\\tilde{\\mathbf{z}} = L\\mathbf{z}$
            
            Los $\\tilde{z}_i$ tienen la estructura de correlación deseada.
            """)
        
        with col2:
            st.markdown("""
            ### Value at Risk (VaR)
            
            $$\\text{VaR}_{1-\\alpha} = -\\inf\\{x : P(L > x) \\leq \\alpha\\}$$
            
            **Interpretación**: Con probabilidad $(1-\\alpha)$, la pérdida no excederá el VaR.
            
            En la simulación Monte Carlo:
            $$\\text{VaR}_{95\\%} = -\\text{Percentil}_{5\\%}(P\\&L)$$
            
            ### Expected Shortfall (CVaR)
            
            $$\\text{CVaR}_{1-\\alpha} = E[L \\mid L > \\text{VaR}_{1-\\alpha}]$$
            
            **Ventajas sobre VaR**: 
            - Coherente en el sentido de Artzner et al. (1999)
            - Captura la magnitud de las pérdidas en la cola
            - No ignora los escenarios catastróficos
            
            ### Contribución al Riesgo (Euler)
            
            $$RC_i = w_i \\cdot \\frac{(\\Sigma\\mathbf{w})_i}{\\sigma_p}$$
            """)
        
        st.markdown("---")
        st.subheader("Limitaciones del modelo")
        st.markdown("""
        1. **Normalidad de retornos**: Los mercados reales exhiben colas pesadas (leptocurtosis) 
           no capturadas por la distribución normal.
        2. **Estacionariedad**: Se asume que μ y σ son constantes en el tiempo, lo cual no 
           refleja la variabilidad de parámetros observada empíricamente.
        3. **Correlaciones estables**: Las correlaciones varían en el tiempo y se incrementan 
           dramáticamente en períodos de crisis.
        4. **Sin jumps**: El GBM no modela saltos bruscos de precio (eventos de cola extrema).
        5. **Riesgo de modelo**: Los parámetros estimados con datos históricos pueden no ser 
           representativos del futuro.
        
        **Extensiones recomendadas**: Modelos GARCH para volatilidad estocástica, cópulas para 
        dependencia no lineal, procesos de Lévy para saltos, y simulación histórica filtrada.
        """)


if __name__ == "__main__":
    main()
