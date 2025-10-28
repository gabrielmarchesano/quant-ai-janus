# janusv3.py
# - Lê `acoes.csv` (lista de tickers) e `df_todas_acoes.csv` (histórico completo, separador ";")
# - Calcula indicadores técnicos (pandas_ta, com fallback manual para os principais)
# - Classifica regime (tendência / lateral / indecisão) de forma automática
# - Calcula score híbrido e executa backtest com controle de risco baseado em ATR
# - Imprime progresso (como o v2) e um resumo de performance ao final
#
# Requisitos principais:
#   pip install pandas numpy pandas_ta (opcional)  # pandas_ta é opcional (há fallback parcial)
#
# Estratégia: Score adaptativo que escolhe automaticamente entre
# tendência (momentum) e reversão à média, com gestão de risco por ATR.
#
# Execução (exemplos):
#   python janusV3_compat.py --capital 100000 --inicio 2010-01-01 --fim 2025-10-01
#   python janusV3_compat.py --csv_acoes acoes.csv --capital 150000
#
# Saídas:
#   resultados_portfolio_janusv3_compat.csv
#   resultados_trades_janusv3_compat.csv

import argparse
import math
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Importa o módulo de estatísticas do repositório
# (ele já lê df_todas_acoes.csv e calcula um DataFrame MultiIndex (Date, Ticker))
import estatisticas as est


# =========================
# Utilidades de leitura
# =========================

def ler_lista_acoes(caminho_csv_acoes: str) -> list:
    """Lê a lista de tickers de `acoes.csv` (coluna 'Acoes')."""
    if not os.path.exists(caminho_csv_acoes):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_csv_acoes}")
    # tenta com ; depois com ,
    try:
        df = pd.read_csv(caminho_csv_acoes, sep=';')
    except Exception:
        df = pd.read_csv(caminho_csv_acoes, sep=',')
    col = 'Acoes' if 'Acoes' in df.columns else df.columns[0]
    return [str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()]


def carregar_base_mercado(filtro_tickers: list | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega dados brutos de mercado (df_mercado do módulo estatisticas) e
    calcula o DF de indicadores com `calcular_estatisticas`.
    Se filtro_tickers for fornecido, aplica no MultiIndex ('Ticker').
    Retorna: (df_mercado_filtrado, df_indicadores_com_precos)
    """
    # df_mercado vem do estatisticas.py (MultiIndex Date,Ticker)
    df_mercado = est.df_mercado.copy()

    if filtro_tickers:
        df_mercado = df_mercado[df_mercado.index.get_level_values('Ticker').isin(filtro_tickers)]

    # Calcula estatísticas (MultiIndex)
    df_ind = est.calcular_estatisticas(df_mercado)

    # Garante presença das colunas de preço no DF final unindo com df_mercado
    # (est.calcular_estatisticas retorna apenas colunas de indicadores)
    df_final = df_ind.join(df_mercado[['Open', 'High', 'Low', 'Close', 'Volume']], how='left')

    # Normaliza nomes que podem vir com variação de caixa/underscore
    # Volatilidade anualizada pode aparecer como 'Vol_anualizadas_21' ou 'Vol_Anualizada_21'
    if 'Vol_anualizadas_21' in df_final.columns and 'Vol_Anualizada_21' not in df_final.columns:
        df_final.rename(columns={'Vol_anualizadas_21': 'Vol_Anualizada_21'}, inplace=True)

    return df_mercado, df_final.dropna()


# =========================
# Parâmetros & Carteira
# =========================

@dataclass
class ParametrosJanusCompat:
    capital_inicial: float = 100000.0
    risco_por_trade_pct: float = 0.01
    multiplicador_stop_atr: float = 2.0
    max_posicoes: int = 5
    adx_tendencia: float = 25.0
    adx_lateral: float = 20.0
    limiar_compra: float = 70.0
    limiar_manutencao: float = 50.0
    custo_transacao: float = 0.0003  # 3 bps


class CarteiraJanusCompat:
    """
    Backtest estilo janusV1, mas com score adaptativo usando os
    indicadores do `estatisticas.py`:
      - Regime tendência: ADX alto + preço acima da média longa (150)
      - Regime lateral: ADX baixo ou preço próximo da média
      - Score momentum (ADX, MACD_Momento, z/distância da média) vs
        score reversão (RSI, z_score_5)
    """

    def __init__(self, params: ParametrosJanusCompat, df_ind: pd.DataFrame):
        self.p = params
        self.df = df_ind.sort_index()  # MultiIndex (Date, Ticker)
        self.tickers = sorted(self.df.index.get_level_values('Ticker').unique())
        self.risco_trade_reais = self.p.capital_inicial * self.p.risco_por_trade_pct

    # -------------- ferramentas internas --------------

    @staticmethod
    def _clip(v, a, b):
        return float(np.minimum(np.maximum(v, a), b))

    @staticmethod
    def _escala_0a100(x, xmin, xmax, inverter=False):
        if pd.isna(x):
            return 0.0
        if xmax == xmin:
            return 0.0
        z = 100.0 * (x - xmin) / (xmax - xmin)
        z = np.clip(z, 0.0, 100.0)
        return float(100.0 - z) if inverter else float(z)

    def _regime(self, linha: pd.Series) -> str:
        """
        Define regime baseado em:
        - ADX (coluna 'ADX')
        - Preço vs média longa (media_movel_simples_150)
        - Proximidade do preço à média (±1%)
        """
        adx = linha.get('ADX', np.nan)
        close = linha.get('Close', np.nan)
        m150 = linha.get('media_movel_simples_150', np.nan)

        if any(pd.isna(x) for x in [adx, close, m150]) or m150 == 0:
            return 'indecisao'

        dist = abs(close - m150) / m150

        if (adx >= self.p.adx_tendencia) and (close > m150):
            return 'tendencia'
        elif (adx <= self.p.adx_lateral) or (dist <= 0.01):
            return 'lateral'
        else:
            return 'indecisao'

    def _score(self, linha: pd.Series) -> float:
        """
        Score 0..100 por linha (Date,Ticker), adaptando pesos ao regime.
        Usa colunas providas por estatisticas.py:
          - 'ADX', 'MACD_Momento', 'RSI_14', 'z_score_5', 'Close',
            'media_movel_simples_150'
        """
        regime = self._regime(linha)
        adx = linha.get('ADX', np.nan)
        macd_mom = linha.get('MACD_Momento', np.nan)
        rsi = linha.get('RSI_14', np.nan)
        z5 = linha.get('z_score_5', np.nan)
        close = linha.get('Close', np.nan)
        m150 = linha.get('media_movel_simples_150', np.nan)

        score = 0.0

        if regime == 'tendencia' and close > m150:
            # Força de tendência (ADX 25..50) + momentum (MACD_Momento > 0)
            s_adx = self._escala_0a100(adx, 25.0, 50.0)
            # MACD_Momento: normaliza por IQR local para robustez
            mm = macd_mom
            if pd.isna(mm):
                s_macd = 0.0
            else:
                # Limites heurísticos: percentil robusto
                s_macd = self._escala_0a100(mm, 0.0, np.nanpercentile(self.df['MACD_Momento'].dropna(), 75))
            score = 0.65 * s_adx + 0.35 * s_macd

        elif regime == 'lateral':
            # Reversão: RSI baixo e z-score curto negativo
            s_rsi = self._escala_0a100(rsi, 20.0, 45.0, inverter=True)  # 20 => 100; 45 => 0
            # z-score: quanto mais negativo (abaixo de 0), maior a atratividade (cap em -2..0)
            if pd.isna(z5):
                s_z = 0.0
            else:
                s_z = self._escala_0a100(z5, -2.0, 0.0, inverter=True)
            score = 0.7 * s_rsi + 0.3 * s_z

        else:
            score = 0.0

        return float(np.clip(score, 0.0, 100.0))

    def _tamanho_posicao(self, atr: float, preco: float, caixa_disp: float) -> float:
        """Sizing por risco/ATR, com ajuste por caixa disponível."""
        if any([pd.isna(atr), pd.isna(preco)]) or atr <= 0 or preco <= 0:
            return 0.0
        risco_por_acao = atr * self.p.multiplicador_stop_atr
        if risco_por_acao <= 0:
            return 0.0
        qtd_risco = self.risco_trade_reais / risco_por_acao
        custo = qtd_risco * preco
        if custo > caixa_disp:
            return max(0.0, caixa_disp / preco)
        return max(0.0, qtd_risco)

    # -------------- backtest --------------

    def backtest(self, inicio: str | None = None, fim: str | None = None):
        print("\n" + "="*70)
        print("INICIANDO BACKTEST - JANUS V3 (Compat)")
        print("="*70)

        # Agenda de datas úteis presentes no DF (nível 'Date')
        todas_datas = sorted(self.df.index.get_level_values('Date').unique())
        if inicio:
            todas_datas = [d for d in todas_datas if d >= pd.Timestamp(inicio)]
        if fim:
            todas_datas = [d for d in todas_datas if d <= pd.Timestamp(fim)]
        if not todas_datas:
            print("Não há datas no intervalo especificado.")
            return None, None

        print(f"Período: {todas_datas[0].date()} a {todas_datas[-1].date()}")
        print(f"Total de dias: {len(todas_datas)}\n")

        caixa = self.p.capital_inicial
        posicoes = {}  # ticker -> dict(...)
        hist_port = []
        hist_trades = []

        for i, data in enumerate(todas_datas):
            if i % 100 == 0:
                print(f"Processando dia {i+1}/{len(todas_datas)}...")

            # 1) Atualiza valor das posições
            valor_total = caixa
            for tck in list(posicoes.keys()):
                try:
                    linha = self.df.loc[(data, tck)]
                except KeyError:
                    continue
                preco = float(linha['Close'])
                pos = posicoes[tck]
                pos['preco_atual'] = preco
                pos['valor_atual'] = pos['qtd'] * preco
                pos['retorno'] = (preco / pos['preco_entrada']) - 1.0
                valor_total += pos['valor_atual']

            # 2) Scores do dia (para todos os tickers com dados nessa data)
            sub = self.df.loc[data] if data in self.df.index.get_level_values('Date') else None
            scores = []
            if sub is not None is not False:
                for tck, linha in sub.iterrows():
                    s = self._score(linha)
                    scores.append({
                        'ticker': tck,
                        'score': s,
                        'preco': float(linha['Close']),
                        'atr': float(linha.get('ATR', np.nan)) if not pd.isna(linha.get('ATR', np.nan)) else 0.0
                    })
            scores = sorted(scores, key=lambda x: x['score'], reverse=True)
            mapa_scores = {s['ticker']: s['score'] for s in scores}

            # 3) Fechamentos por stop/sinal
            for tck in list(posicoes.keys()):
                if (data, tck) not in self.df.index:
                    continue
                linha = self.df.loc[(data, tck)]
                preco_atual = float(linha['Close'])
                pos = posicoes[tck]
                fechar = False
                if preco_atual < pos['stop_loss']:
                    fechar = True
                elif (tck not in mapa_scores) or (mapa_scores[tck] < self.p.limiar_manutencao):
                    fechar = True

                if fechar:
                    saida = preco_atual
                    proventos = pos['qtd'] * saida
                    custo = proventos * self.p.custo_transacao
                    caixa += proventos - custo

                    ret = (saida / pos['preco_entrada']) - 1.0
                    hist_trades.append({
                        'ticker': tck,
                        'entry_date': pos['data_entrada'],
                        'exit_date': data,
                        'entry_price': pos['preco_entrada'],
                        'exit_price': saida,
                        'shares': pos['qtd'],
                        'return': ret
                    })
                    del posicoes[tck]

            # 4) Aberturas novas (maiores scores acima do limiar)
            caixa_disp = caixa * 0.95
            candidatos = [c for c in scores if (c['score'] >= self.p.limiar_compra) and (c['ticker'] not in posicoes)]
            for cand in candidatos:
                if len(posicoes) >= self.p.max_posicoes:
                    break
                qtd = self._tamanho_posicao(cand['atr'], cand['preco'], caixa_disp)
                if qtd <= 0:
                    continue
                custo_total = qtd * cand['preco']
                if custo_total > caixa:
                    continue
                custo = custo_total * self.p.custo_transacao
                caixa -= (custo_total + custo)
                stop = cand['preco'] - (cand['atr'] * self.p.multiplicador_stop_atr)
                posicoes[cand['ticker']] = {
                    'qtd': qtd,
                    'preco_entrada': cand['preco'],
                    'data_entrada': data,
                    'stop_loss': stop,
                    'preco_atual': cand['preco'],
                    'valor_atual': custo_total,
                    'retorno': 0.0
                }
                caixa_disp -= custo_total

            # 5) Registro diário
            hist_port.append({
                'date': data,
                'total_value': valor_total,
                'cash': caixa,
                'n_positions': len(posicoes)
            })

        df_port = pd.DataFrame(hist_port).set_index('date')
        df_trades = pd.DataFrame(hist_trades)

        self._print_resultados(df_port, df_trades)
        return df_port, df_trades

    # -------------- impressão de resultados --------------

    def _print_resultados(self, df_port: pd.DataFrame, df_trades: pd.DataFrame):
        final_value = float(df_port['total_value'].iloc[-1])
        total_return = (final_value / self.p.capital_inicial) - 1.0

        df_port = df_port.copy()
        df_port['daily_return'] = df_port['total_value'].pct_change().fillna(0.0)

        dias = len(df_port)
        anos = dias / 252.0 if dias > 0 else 0.0
        cagr = (final_value / self.p.capital_inicial) ** (1.0 / anos) - 1.0 if anos > 0 else 0.0

        vol_anual = df_port['daily_return'].std() * math.sqrt(252.0)
        sharpe = (df_port['daily_return'].mean() * 252.0) / vol_anual if vol_anual > 0 else 0.0

        acumulado = (1.0 + df_port['daily_return']).cumprod()
        pico = acumulado.cummax()
        drawdown = (acumulado - pico) / pico
        max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

        if df_trades is not None and not df_trades.empty:
            win_rate = float((df_trades['return'] > 0).mean())
            avg_win = float(df_trades.loc[df_trades['return'] > 0, 'return'].mean()) if (df_trades['return'] > 0).any() else 0.0
            avg_loss = float(df_trades.loc[df_trades['return'] < 0, 'return'].mean()) if (df_trades['return'] < 0).any() else 0.0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
            n_trades = len(df_trades)
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0.0
            n_trades = 0

        print("\n" + "="*70)
        print("RESULTADOS DO BACKTEST (JANUS V3 Compat)")
        print("="*70)
        print(f"\nCAPITAL:")
        print(f"  Inicial: R$ {self.p.capital_inicial:,.2f}")
        print(f"  Final: R$ {final_value:,.2f}")
        print(f"  Lucro: R$ {final_value - self.p.capital_inicial:,.2f}")
        print(f"\nRETORNO:")
        print(f"  Total: {total_return*100:.2f}%")
        print(f"  CAGR: {cagr*100:.2f}% ao ano")
        print(f"\nRISCO:")
        print(f"  Volatilidade: {vol_anual*100:.2f}%")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Max Drawdown: {max_dd*100:.2f}%")
        print(f"\nTRADES:")
        print(f"  Total: {n_trades}")
        print(f"  Win Rate: {win_rate*100:.1f}%")
        print(f"  Retorno médio (ganho): {avg_win*100:.2f}%")
        print(f"  Retorno médio (perda): {avg_loss*100:.2f}%")
        print(f"  Profit Factor: {profit_factor:.2f}")
        print("="*70)


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(description="JANUS V3 compat - backtest com score adaptativo (usa estatisticas.py).")
    ap.add_argument("--csv_acoes", type=str, default=None, help="(Opcional) CSV com coluna 'Acoes' para filtrar tickers.")
    ap.add_argument("--capital", type=float, default=100000.0)
    ap.add_argument("--inicio", type=str, default=None)
    ap.add_argument("--fim", type=str, default=None)
    ap.add_argument("--max_pos", type=int, default=5)
    ap.add_argument("--risco_trade", type=float, default=0.01)
    ap.add_argument("--stop_atr", type=float, default=2.0)
    ap.add_argument("--compra", type=float, default=70.0)
    ap.add_argument("--manter", type=float, default=50.0)
    ap.add_argument("--adx_tend", type=float, default=25.0)
    ap.add_argument("--adx_lat", type=float, default=20.0)
    ap.add_argument("--custo", type=float, default=0.0003)

    args = ap.parse_args()

    # Carrega tickers (se fornecido)
    tickers = None
    if args.csv_acoes:
        try:
            tickers = ler_lista_acoes(args.csv_acoes)
            print(f"✓ {len(tickers)} tickers carregados de {args.csv_acoes}.")
        except Exception as e:
            print(f"Erro ao ler {args.csv_acoes}: {e}")
            return 1

    # Carrega base e indicadores
    df_mercado, df_ind = carregar_base_mercado(tickers)
    print(f"✓ Base carregada: {len(df_mercado)} linhas de mercado; {len(df_ind)} com indicadores.\n")

    params = ParametrosJanusCompat(
        capital_inicial=args.capital,
        risco_por_trade_pct=args.risco_trade,
        multiplicador_stop_atr=args.stop_atr,
        max_posicoes=args.max_pos,
        adx_tendencia=args.adx_tend,
        adx_lateral=args.adx_lat,
        limiar_compra=args.compra,
        limiar_manutencao=args.manter,
        custo_transacao=args.custo
    )

    carteira = CarteiraJanusCompat(params, df_ind)
    df_port, df_trades = carteira.backtest(inicio=args.inicio, fim=args.fim)

    # Salva resultados
    if df_port is not None and not df_port.empty:
        df_port.to_csv("resultados_portfolio_janusv3_compat.csv")
        print("Resultados do portfólio salvos em: resultados_portfolio_janusv3_compat.csv")
    if df_trades is not None and not df_trades.empty:
        df_trades.to_csv("resultados_trades_janusv3_compat.csv", index=False)
        print("Trades salvos em: resultados_trades_janusv3_compat.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
