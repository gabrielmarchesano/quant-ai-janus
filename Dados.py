import yfinance as yf
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


df_acoes = pd.read_csv("acoes.csv")



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
            df_csv_filtrados = df_valores_acao[["Date","Open","High","Low","Close","Volume"]].copy()
            df_csv_filtrados.to_csv(f"{acao}.csv",index=False,sep = ";")

    except Exception as e:
        print(f"❌ Erro ao buscar dados de {acao}: {e}")

