import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller,acf
import matplotlib.pyplot as plt


df_mercado = pd.read_csv("df_todas_acoes.csv",index_col=["Date","Ticker"],sep=";", parse_dates=True)



def calcular_volatilidade(df, janela_dias=21, dias_uteis_ano=252):
    
    # 1. Calcular Retornos Logarítmicos
    df['Retornos_Log'] = df.groupby(level='Ticker')['Close'].apply(lambda x: np.log(x / x.shift(1))).droplevel(0)
    
    # 2. Calcular o Desvio Padrão Diário Móvel (Volatilidade Diária)
    # Multiplicamos pela raiz de N para anualizar (N=252 é o número de dias úteis)
    fator_anualizacao = np.sqrt(dias_uteis_ano)
    
    df[f'Vol_Anualizada_{janela_dias}'] = df.groupby(level='Ticker')['Retornos_Log'].rolling(window=janela_dias).std().droplevel(0) * fator_anualizacao
    
    return df


def testar_estacionaridade_adf(serie_precos):
    """Calcula os retornos e aplica o teste ADF, retornando o p-valor e a estatística."""
    
    # Gerar a série de retornos
    # dropna() é crucial, pois o primeiro valor é NaN
    serie_retornos = serie_precos.pct_change().dropna()
    
    # O teste ADF requer pelo menos 50 observações, mas vamos apenas checar
    if len(serie_retornos) < 10:
        return pd.Series([None, None], index=['ADF Statistic', 'p-value'])
    
    # Aplicar o teste ADF
    # regression='c' inclui apenas constante (intercepto)
    resultado = adfuller(serie_retornos, regression='c')
    
    # Retornar o resultado em formato de Série para fácil consolidação
    return pd.Series([resultado[0], resultado[1]], index=['ADF Statistic', 'p-value'])


def analisar_autocorrelacao(serie_retornos, ticker, lags=20):
    """Gera os gráficos ACF e PACF para análise visual."""
    serie_limpa = serie_retornos.dropna()
    
    if len(serie_limpa) < lags:
        print(f"Dados insuficientes para {ticker}")
        return

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    # ACF - Autocorrelação Total
    plot_acf(serie_limpa, lags=lags, ax=ax[0], title=f'ACF Retornos Log - {ticker}')
    
    # PACF - Autocorrelação Parcial (eliminando a influência dos lags intermediários)
    plot_pacf(serie_limpa, lags=lags, ax=ax[1], title=f'PACF Retornos Log - {ticker}')
    
    plt.tight_layout()
    plt.show()

def calcular_breakout(df, janela=20):
    
    grupos = df.groupby(level='Ticker')

    # 1. Definir o Canal (Suporte/Resistência)
    df[f'Max_{janela}'] = grupos['High'].rolling(window=janela).max().shift(1).droplevel(0)
    df[f'Min_{janela}'] = grupos['Low'].rolling(window=janela).min().shift(1).droplevel(0)

    # 2. Identificar Breakouts
    
    # Breakout de Compra (Bullish): Fechamento acima da Máxima anterior
    df[f'Breakout_Compra_{janela}'] = np.where(df['Close'] > df[f'Max_{janela}'], 1, 0)
    
    # Breakout de Venda (Bearish): Fechamento abaixo da Mínima anterior
    df[f'Breakout_Venda_{janela}'] = np.where(df['Close'] < df[f'Min_{janela}'], -1, 0)
    
    # Combina os sinais (1: Compra, -1: Venda, 0: Nenhum)
    df['Sinal_Breakout'] = df[f'Breakout_Compra_{janela}'] + df[f'Breakout_Venda_{janela}']
    
    # Limpa colunas auxiliares, se desejar
    df = df.drop(columns=[f'Max_{janela}', f'Min_{janela}', f'Breakout_Compra_{janela}', f'Breakout_Venda_{janela}'])

    return df

def calcular_adx_completo(df, janela=14):
    
    df_result = df.copy()
    grupos = df_result.groupby(level="Ticker")
    
    # --- 1. True Range (TR) ---
    # Usamos as Series diretas
    high = df_result['High'] 
    low = df_result['Low']
    close = df_result['Close']
    
    tr1 = high - low
    
    # Usamos grupos['Close'].shift(1) para garantir que o shift seja por ticker
    tr2 = (high - grupos['Close'].shift(1)).abs() 
    tr3 = (low - grupos['Close'].shift(1)).abs()
    
    # O .max(axis=1) em um concat de Series com o mesmo índice preserva o MultiIndex
    df_result['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1) 

    # --- 2. Directional Movement (+DM e -DM) ---
    
    # O .diff() aplicado ao SeriesGroupBy garante a diferença dentro de cada grupo
    up_move = grupos['High'].diff() 
    down_move = -grupos['Low'].diff()

    # O np.where trabalha com Series que têm o mesmo índice (MultiIndex)
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    df_result['+DM'] = pd.Series(pos_dm.ravel(), index=df_result.index) # Use ravel() para garantir a dimensão
    
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    df_result['-DM'] = pd.Series(neg_dm.ravel(), index=df_result.index)

    # --- 3. Smoothed TR, +DM, -DM (Usando a suavização de Wilder) ---
    
    def wilder_smoothing(series, window):
        # A EWM aplicada ao groupby gera um MultiIndex de 3 níveis. 
        # O .droplevel(0) remove o nível extra e restaura o (Date, Ticker)
        return series.groupby(level='Ticker').apply(lambda x: x.ewm(
            alpha=1/window, adjust=False).mean()
        ).droplevel(0) # CORREÇÃO CRUCIAL (como discutido no problema anterior)

    # Cálculo dos valores suavizados
    df_result['Smoothed_TR'] = wilder_smoothing(df_result['TR'], janela)
    df_result['Smoothed_+DM'] = wilder_smoothing(df_result['+DM'], janela)
    df_result['Smoothed_-DM'] = wilder_smoothing(df_result['-DM'], janela)
    
    df_result['ATR'] = df_result['Smoothed_TR']

    # --- 4. Directional Indicators (+DI e -DI) ---
    
    # O alinhamento funciona corretamente com o MultiIndex (Date, Ticker)
    df_result['+DI'] = (df_result['Smoothed_+DM'] / df_result['Smoothed_TR']) * 100
    df_result['-DI'] = (df_result['Smoothed_-DM'] / df_result['Smoothed_TR']) * 100

    # --- 5. Directional Index (DX) e 6. ADX ---
    
    di_diff = (df_result['+DI'] - df_result['-DI']).abs()
    di_sum = df_result['+DI'] + df_result['-DI']
    
    df_result['DX'] = (di_diff / di_sum).replace([np.inf, -np.inf], 0).fillna(0) * 100

    df_result['ADX'] = wilder_smoothing(df_result['DX'], janela)

    # Limpar colunas auxiliares
    colunas_para_remover = ['TR', '+DM', '-DM', 'Smoothed_TR', 'Smoothed_+DM', 'Smoothed_-DM', 'DX']
    df_result = df_result.drop(columns=colunas_para_remover, errors='ignore')

    return df_result

def calcular_estatisticas(df):
    df_estatisticas = pd.DataFrame(index = df.index)

    close_precos = df["Close"]
    grupos_por_ticker = close_precos.groupby(level="Ticker")
    #rpz pelo q vi aq nas estatisticas é meio livre ent tem q ir testando e usasr as que derem mais certo


    #valor para window da media movel pode ser alterado, usar uma media movel com window grande, para pegar o movimento sem os ruidos do curto prazo (tendencia)
    # e uma pequena para pegar essas variacoes curtas(reversao)
    #-----------------------------------estatisticas simples ----------------------------------------------------------------------------------------------------
    janela_curta = 5
    janela_media = 45
    janela_longa = 150

    df_estatisticas[f"media_movel_simples_{janela_curta}"] = grupos_por_ticker.rolling(window=janela_curta).mean().droplevel(0)
    df_estatisticas[f"media_movel_simples_{janela_longa}"] = grupos_por_ticker.rolling(window=janela_longa).mean().droplevel(0)
    df_estatisticas[f"media_movel_simples_{janela_media}"] = grupos_por_ticker.rolling(window=janela_media).mean().droplevel(0)
    df_estatisticas[f"media_movel_exponencial_{janela_curta}"] = grupos_por_ticker.ewm(span=janela_curta, adjust=False).mean().droplevel(0)
    df_estatisticas[f"media_movel_exponencial_{janela_longa}"] = grupos_por_ticker.ewm(span=janela_longa, adjust=False).mean().droplevel(0)
    df_estatisticas[f"media_movel_exponencial_{janela_media}"] = grupos_por_ticker.ewm(span=janela_media, adjust=False).mean().droplevel(0)

    media_curta = df_estatisticas[f"media_movel_simples_{janela_curta}"]
    desvio_padrao_curto = grupos_por_ticker.rolling(window=janela_curta).std().droplevel(0)
    df_estatisticas[f"desvio_padrao_{janela_curta}"] = desvio_padrao_curto

    df_estatisticas[f"z_score_{janela_curta}"] = (close_precos - media_curta)/desvio_padrao_curto


    media_longa = df_estatisticas[f"media_movel_simples_{janela_longa}"]
    desvio_padrao_longo = grupos_por_ticker.rolling(window=janela_longa).std().droplevel(0)
    df_estatisticas[f"desvio_padrao_{janela_longa}"] = desvio_padrao_longo

    df_estatisticas[f"z_score_{janela_longa}"] = (close_precos - media_longa)/desvio_padrao_longo

     #-----------------------------------estatisticas mais complexas ----------------------------------------------------------------------------------------------------

    def ema(serie,span):
        return serie.groupby(level="Ticker").ewm(span=span,adjust=False).mean().droplevel(0)

    janela_rsi = 14
    janela_macd_curta = 12
    janela_macd_longa = 26
    janela_macd_sinal = 9
    janela_adx = 14 # ou 20, dependendo da estratégia

     # --------------------------------------------------------------------------

    #rsi 
    delta = grupos_por_ticker.diff()
    ganho = delta.where(delta > 0, 0)
    perda = -delta.where(delta < 0, 0)
    

    avg_ganho = ganho.groupby(level="Ticker").ewm(span=janela_rsi, adjust=False).mean().droplevel(0)
    avg_perda = perda.groupby(level="Ticker").ewm(span=janela_rsi, adjust=False).mean().droplevel(0)

   
    rs = avg_ganho / avg_perda
    
   
    df_estatisticas[f"RSI_{janela_rsi}"] = 100 - (100 / (1 + rs))

    #MACD

    ema_curta = ema(close_precos, janela_macd_curta)
    ema_longa = ema(close_precos, janela_macd_longa)
    df_estatisticas["MACD_Linha"] = ema_curta - ema_longa

    macd_linh_serie = df_estatisticas["MACD_Linha"]
    df_estatisticas["MACD_Sinal"] = macd_linh_serie.groupby(level="Ticker").ewm(span=janela_macd_sinal, adjust=False).mean().droplevel(0)

    df_estatisticas["MACD_Momento"] = df_estatisticas["MACD_Linha"] - df_estatisticas["MACD_Sinal"]


    #ADX

    df_adx = calcular_adx_completo(df_mercado)

    df_estatisticas["ADX"] = df_adx["ADX"]
    df_estatisticas["-DI"] = df_adx["-DI"]
    df_estatisticas["+DI"] = df_adx["+DI"]
    df_estatisticas["ATR"] = df_adx["ATR"] 

    #volatilidade
    df_volatilidade = calcular_volatilidade(df_mercado)
    df_estatisticas["Vol_anualizadas_21"] = df_volatilidade["Vol_Anualizada_21"]

    #breakout

    df_breakout = calcular_breakout(df_mercado)
    df_estatisticas[f'Sinal_Breakout'] = df_breakout['Sinal_Breakout']

    #estacionaridades

    df_estacionaridade = (
        df_mercado.
        groupby("Ticker")["Close"].
        apply(testar_estacionaridade_adf)
    )
    

    df_resultados_adf = df_estacionaridade.unstack()

    df_resultados_adf.index.name = 'Ticker' 

    significancia = 0.05
    df_resultados_adf['Estacionario_ADF'] = np.where(df_resultados_adf['p-value'] < significancia, 1, 0)
    
    # Renomear as colunas para evitar conflitos e deixar claro o que é
    df_resultados_adf = df_resultados_adf.rename(columns={
        'ADF Statistic': 'ADF_Statistic_Global',
        'p-value': 'ADF_p_value_Global'
    })

    # --- 4. MERGE (A chave de união é o Ticker) ---

    # Para usar o merge, precisamos que 'Ticker' seja uma coluna (não parte do MultiIndex)
    # ou que o merge seja feito com o índice de df_estatisticas (o Ticker)

    # Resetamos o índice do df_estatisticas temporariamente para usar o 'Ticker' como coluna para o merge
    df_estatisticas_temp = df_estatisticas.reset_index(level='Date') # Apenas mantém o Ticker como índice para o merge

    # Usamos o merge, juntando por Ticker (que é o Index em ambos os DFs após a transformação)
    df_estatisticas = df_estatisticas_temp.merge(
        df_resultados_adf, 
        how='left', 
        left_index=True,  # Junta pelo Ticker (Index) do df_estatisticas_temp
        right_index=True  # Junta pelo Ticker (Index) do df_resultados_adf
    )
    
    # 5. Restaurar o MultiIndex original (Date, Ticker)
    df_estatisticas = df_estatisticas.set_index('Date', append=True).reorder_levels(['Date', 'Ticker'])

    return df_estatisticas

df_estat = calcular_estatisticas(df_mercado)


print(df_estat.tail(230))
  