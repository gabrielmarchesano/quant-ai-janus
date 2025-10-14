import pandas as pd


df_mercado = pd.read_csv("df_todas_acoes.csv",index_col=["Date","Ticker"],sep=";", parse_dates=True)



def calcular_estatisticas(df):
    df_estatisticas = pd.DataFrame(index = df.index)

    close_precos = df["Close"]
    grupos_por_ticker = close_precos.groupby(level="Ticker")
    #rpz pelo q vi aq nas estatisticas é meio livre ent tem q ir testando e usasr as que derem mais certo


    #valor para window da media movel pode ser alterado, usar uma media movel com window grande, para pegar o movimento sem os ruidos do curto prazo (tendencia)
    # e uma pequena para pegar essas variacoes curtas(reversao)
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

    return df_estatisticas

df_estat = calcular_estatisticas(df_mercado)


print(df_estat.tail(50))
  