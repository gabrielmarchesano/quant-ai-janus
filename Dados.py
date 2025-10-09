import yfinance as yf
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df_acoes = pd.read_csv("acoes.csv")

lista_de_dfs = []

for indice,linha_acao in df_acoes.iterrows():
    try:
        acao = linha_acao["Acoes"].strip()
        print(f"\n📊 Baixando dados de: {acao}")
        tk = yf.Ticker(acao)

        dados = tk.history(period="max")
        df_com_data = dados.reset_index()
       
        if dados.empty:
            print("⚠️ Dados não encontrados.")
        else:
            df_valores_acao = pd.DataFrame(df_com_data)
            df_valores_acao['Ticker'] = acao
            df_csv_filtrados = df_valores_acao[["Date","Ticker","Open","High","Low","Close","Volume"]].copy()
            lista_de_dfs.append(df_csv_filtrados)

    except Exception as e:
        print(f"❌ Erro ao buscar dados de {acao}: {e}")

if lista_de_dfs:
    df_unico = pd.concat(lista_de_dfs, ignore_index=True)
    df_backtest_final = df_unico.sort_values(by=['Date', 'Ticker'])
    df_backtest_final.to_csv(
        "df_todas_acoes.csv", 
        index=False, 
        sep=';', 
    )
    print(f"✅ Sucesso! Todos os dados foram salvos e ordenados em: {nome_arquivo_final}")
    print(f"Total de linhas (registros diários) no arquivo final: {len(df_backtest_final)}")



