import pandas as pd
import numpy as np 
import math
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import yfinance as yf
import scipy.stats as stats
from scipy.stats import kurtosis, skew, shapiro, norm, t

st.title('PROYECTO 1')
st.subheader('Santiago Yael Morales Torres')

### a) CARGAR LOS DATOS DESDE YAHOO FINANCE

#Cargamos el dataframe
stock = 'WMT' 
precios = yf.download(stock, start="2010-01-01")['Close']
st.header(f'Evolución del Precio de {stock} desde 2010')

#Mostramos el dataframe y la serie de tiempo del histórico del precio de cierre
st.dataframe(precios.sort_index(ascending=False))
st.subheader('Gráfico de precios')
st.line_chart(precios[stock], x_label='Fecha', y_label='Precio')

### b) RENDIMIENTOS DIARIOS: MEDIA, SESGO Y CURTOSIS

st.header(f'Evolución de los Rendimientos de {stock} desde 2010')

#Obtenemos y visualizamos el dataframe y la serie de tiempo de los rendimientos
rendimientos = precios.pct_change().dropna()
st.dataframe(rendimientos.sort_index(ascending=False))
st.subheader('Gráfico de rendimientos diarios')
st.line_chart(rendimientos[stock], x_label='Fecha', y_label='Rendimiento Diario')

#Histograma de los rendimientos
st.subheader('Histograma y Box-Plot de rendimientos diarios')
fig = px.histogram(rendimientos, 
                   x=stock, 
                   nbins=30,
                   labels={stock:'Rendimiento Diario'}, 
                   marginal="box")
fig.update_layout(yaxis_title="Frecuencia")
st.plotly_chart(fig)

#QQ-plot de los rendimientos
st.subheader('QQplot de rendimientos diarios')
fig, ax = plt.subplots(figsize=(5, 3))
stats.probplot(rendimientos[stock], dist="norm", plot=ax)
ax.set_title("Q-Q Plot de los Rendimientos")
st.pyplot(fig)

#Métricas de los rendimientos
st.subheader(f'Métricas de los Rendimientos: {stock}')
media_rend = rendimientos[stock].mean()
desv_est_rend = rendimientos[stock].std()
curtosis_rend = kurtosis(rendimientos[stock])
sesgo_rend = skew(rendimientos[stock])
col1,col2,col3,col4 = st.columns(4)

col1.metric("Rendimiento Medio Diario", f"{media_rend:.4%}")
col2.metric("Desviación Estándar", f"{desv_est_rend:.4}")
col3.metric("Curtosis",f"{curtosis_rend:.4}")
col4.metric("Sesgo",f"{sesgo_rend:.4}")

### c) VaR y CVaR al 95%,97.5% y 99%

st.header(f'Métricas de Riesgo de {stock}: VaR y CVaR')

n_conf = [0.95,0.975,0.99]
alpha = st.selectbox("Selecciona un nivel de confianza", n_conf)

if alpha:
    #VaR Paramétrico Normal
    VaR_par_norm = norm.ppf(1-alpha,media_rend,desv_est_rend)

    #VaR Paramétrico t-student
    v=5
    VaR_par_t = media_rend + desv_est_rend*math.sqrt((v-2)/v)*t.ppf(1-alpha,v)

    #VaR Histórico
    VaR_hist = rendimientos[stock].quantile(1-alpha)

    #VaR MonteCarlo
    n_sims = 100000
    MC_sim = np.random.normal(media_rend, desv_est_rend, n_sims)
    VaR_MC = np.percentile(MC_sim, (1-alpha)*100)

    #CVaR Paramétrico Normal
    CVaR_par_norm = media_rend - desv_est_rend*norm.pdf(norm.ppf(alpha))/(1-alpha)

    #CVaR Paramétrico t-student
    CVaR_par_t = rendimientos[stock][rendimientos[stock] <= VaR_par_t].mean()
    
    #CVaR Histórico
    CVaR_hist = rendimientos[stock][rendimientos[stock] <= VaR_hist].mean()

    #CVaR MonteCarlo
    CVaR_MC = MC_sim[MC_sim <= VaR_MC].mean()
    
    col5, col6, col7, col8= st.columns(4)
    col5.metric("VaR Paramétrico Normal", f"{VaR_par_norm:.4%}")
    col6.metric("VaR Paramétrico t-student", f"{VaR_par_t:.4%}")
    col7.metric("VaR Histórico", f"{VaR_hist:.4%}")
    col8.metric("VaR Monte Carlo", f"{VaR_MC:.4%}")

    col9, col10, col11, col12= st.columns(4)
    col9.metric("CVaR Paramétrico Normal", f"{CVaR_par_norm:.4%}")
    col10.metric("CVaR Paramétrico t-student", f"{CVaR_par_t:.4%}")
    col11.metric("CVaR Histórico", f"{CVaR_hist:.4%}")
    col12.metric("CVaR Monte Carlo", f"{CVaR_MC:.4%}")

    #Histograma de rendimientos junto con las métricas de riesgo
    st.subheader('Histograma de rendimientos diarios junto con las métricas de riesgo')
    fig = px.histogram(rendimientos, 
                    x=stock, 
                    nbins=30,
                    labels={stock:'Rendimiento Diario'}, 
                    marginal="box")
    fig.update_layout(yaxis_title="Frecuencia")

    #agregar lineas para los VaR's y CVaR
    fig.add_vline(x=VaR_par_norm, line=dict(color='red', dash='dash', width=1.5))
    fig.add_vline(x=VaR_par_t, line=dict(color='orange', dash='dash', width=1.5))
    fig.add_vline(x=VaR_hist, line=dict(color='purple', dash='dash', width=1.5))
    fig.add_vline(x=VaR_MC, line=dict(color='skyblue', dash='dash', width=1.5))
    fig.add_vline(x=CVaR_par_norm, line=dict(color='blue', dash='dash', width=1.5))
    fig.add_vline(x=CVaR_par_t, line=dict(color='green', dash='dash', width=1.5))
    fig.add_vline(x=CVaR_hist, line=dict(color='brown', dash='dash', width=1.5))
    fig.add_vline(x=CVaR_MC, line=dict(color='grey', dash='dash', width=1.5))

    #agregar las leyendas
    fig.add_annotation(x=0.1, y=1500, text=f"Var {alpha*100}% Paramétrico Normal", showarrow=False,
                    font=dict(size=12, color="red"),bgcolor="white",borderpad=4)
    fig.add_annotation(x=0.1, y=1400, text=f"Var {alpha*100}% Paramétrico t-student", showarrow=False,
                    font=dict(size=12, color="orange"),bgcolor="white",borderpad=4)
    fig.add_annotation(x=0.1, y=1300, text=f"Var {alpha*100}% Histórico", showarrow=False,
                    font=dict(size=12, color="purple"),bgcolor="white",borderpad=4)
    fig.add_annotation(x=0.1, y=1200, text=f"Var {alpha*100}% Monte Carlo", showarrow=False,
                    font=dict(size=12, color="skyblue"),bgcolor="white",borderpad=4)
    fig.add_annotation(x=0.1, y=1000, text=f"CVar {alpha*100}% Paramétrico Normal", showarrow=False,
                    font=dict(size=12, color="blue"),bgcolor="white",borderpad=4)
    fig.add_annotation(x=0.1, y=900, text=f"CVar {alpha*100}% Paramétrico t-student", showarrow=False,
                    font=dict(size=12, color="green"),bgcolor="white",borderpad=4)
    fig.add_annotation(x=0.1, y=800, text=f"CVar {alpha*100}% Paramétrico Histórico", showarrow=False,
                    font=dict(size=12, color="brown"),bgcolor="white",borderpad=4)
    fig.add_annotation(x=0.1, y=700, text=f"CVar {alpha*100}% Paramétrico Monte Carlo", showarrow=False,
                    font=dict(size=12, color="grey"),bgcolor="white",borderpad=4)
    
    st.plotly_chart(fig)

### d) Rolling Windows
st.header(f'Métricas de Riesgo de {stock}: VaR y CVaR mediante Rolling Windows')

n_conf_rw = [0.95,0.99]
alpha_rw = st.selectbox("Selecciona un nivel de confianza para las Rolling Windows", n_conf_rw)

if alpha_rw:
    rolling_window = rendimientos.iloc[252:].copy()

    rolling_window['VaR_Parametrico'] = [
        norm.ppf(1-alpha_rw,rendimientos[stock].iloc[i:i+252].mean(),rendimientos[stock].iloc[i:i+252].std()) 
        for i in range(len(rolling_window[stock]))]
    
    rolling_window['VaR_Historico'] = [
        rendimientos[stock].iloc[i:i+252].quantile(1-alpha_rw)
        for i in range(len(rolling_window[stock]))]
    
    rolling_window['CVaR_Parametrico'] = [
        rendimientos[stock].iloc[i:i+252].mean() - rendimientos[stock].iloc[i:i+252].std()*norm.pdf(norm.ppf(alpha_rw))/(1-alpha_rw)
        for i in range(len(rolling_window[stock]))]
    
    rolling_window['CVaR_Historico'] = [
        rendimientos[stock].iloc[i:i+252]
        [rendimientos[stock].iloc[i:i+252] <= rolling_window.iloc[i]['VaR_Historico']].mean()
        for i in range(len(rolling_window[stock]))]
    
    st.dataframe(rolling_window)
    
    #Graficamos las rolling windows
    st.subheader(f'Gráfico de rendimientos diarios y métricas de riesgo al {int(alpha_rw*100)}% de confianza')
    st.line_chart(rolling_window, x_label='Fecha', y_label='Rendimiento Diario')

    ### e) Violaciones al VaR y CVaR

    st.subheader(f'Eficiencia de las métricas de riesgo de {stock}: VaR y CVaR al {int(alpha_rw*100)}% de confianza, \
                 mediante Rolling Windows')

    Violaciones_VaR_parametrico = ((rolling_window[stock] < rolling_window['VaR_Parametrico']).sum())/(len(rolling_window[stock]))
    Violaciones_VaR_historico = ((rolling_window[stock] < rolling_window['VaR_Historico']).sum())/(len(rolling_window[stock]))
    Violaciones_CVaR_parametrico = ((rolling_window[stock] < rolling_window['CVaR_Parametrico']).sum())/(len(rolling_window[stock]))
    Violaciones_CVaR_historico = ((rolling_window[stock] < rolling_window['CVaR_Historico']).sum())/(len(rolling_window[stock]))

    col13, col14, col15, col16= st.columns(4)

    col13.metric("Violaciones VaR Param.", f"{Violaciones_VaR_parametrico:.4%}")
    col14.metric("Violaciones VaR Histórico", f"{Violaciones_VaR_historico:.4%}")
    col15.metric("Violaciones CVaR Param.", f"{Violaciones_CVaR_parametrico:.4%}")
    col16.metric("Violaciones CVaR Histórico", f"{Violaciones_CVaR_historico:.4%}")

### f) Rolling Windows con volatilidad móvil

st.header(f'VaR con volatilidad móvil de {stock}, mediante Rolling Windows')

n_conf_rwm = [0.95,0.99]
alpha_rwm = st.selectbox("Selecciona un nivel de confianza para las Rolling Windows \
                         con volatilidad móvil", n_conf_rwm)

if alpha_rwm:
    rolling_window_movil = rendimientos.iloc[252:].copy()
    rolling_window_movil['VaR_volatilidad_movil'] = [
        norm.ppf(1-alpha_rwm,0,1)*rendimientos[stock].iloc[i:i+252].std() 
        for i in range(len(rolling_window[stock]))]
    st.dataframe(rolling_window_movil)
    
    #Graficamos las rolling windows y el VaR con volatilidad móvil
    st.subheader(f'Gráfico de rendimientos diarios y VaR con volatilidad móvil al {int(alpha_rwm*100)}% de confianza')
    st.line_chart(rolling_window_movil, x_label='Fecha', y_label='Rendimiento Diario')

    #Violaciones al VaR con volatilidad móvil
    st.subheader(f'Eficiencia del VaR con volatilidad móvil al {int(alpha_rwm*100)}% de confianza')

    Violaciones_VaR_vm = ((rolling_window_movil[stock] < rolling_window_movil['VaR_volatilidad_movil']).sum())/(len(rolling_window_movil[stock]))
    col17, col18 = st.columns(2)
    col17.metric("Violaciones VaR con volatilidad móvil", f"{Violaciones_VaR_vm:.4%}")


