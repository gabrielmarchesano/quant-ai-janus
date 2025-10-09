"""
Sistema de Portfolio Trading Quantitativo - Top 30 B3
Operação: Diária (decisão no início do dia)
Backtest: 5 anos históricos
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

TOP30_B3 = [
    'PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'BBAS3',
    'ABEV3', 'WEGE3', 'RENT3', 'SUZB3', 'ELET3',
    'JBSS3', 'RDOR3', 'RAIL3', 'BBSE3', 'LREN3',
    'MGLU3', 'HAPV3', 'PRIO3', 'GGBR4', 'EMBR3',
    'VBBR3', 'BRFS3', 'RADL3', 'ASAI3', 'CSAN3',
    'SANB11', 'CMIG4', 'UGPA3', 'CPLE6', 'VIVT3'
]

class TechnicalIndicators:
    """
    Classe para cálculo de indicadores técnicos
    
    EXPLICAÇÃO DOS INDICADORES:
    
    1. MÉDIAS MÓVEIS (SMA/EMA):
       - Suavizam o preço para identificar tendência
       - SMA: média aritmética simples
       - EMA: dá mais peso aos preços recentes
       - Uso: Cruzamento de médias indica mudança de tendência
    
    2. RSI (Relative Strength Index):
       - Oscilador de momentum (0-100)
       - < 30: ativo sobrevendido (possível compra)
       - > 70: ativo sobrecomprado (possível venda)
       - Mede força relativa de ganhos vs perdas
    
    3. BOLLINGER BANDS:
       - Bandas de volatilidade (média ± 2 desvios padrão)
       - Preço na banda inferior: possível compra
       - Preço na banda superior: possível venda
       - Identifica períodos de alta/baixa volatilidade
    
    4. MACD (Moving Average Convergence Divergence):
       - Indicador de momentum e tendência
       - MACD > Signal: momentum positivo (compra)
       - MACD < Signal: momentum negativo (venda)
       - Histograma mostra força do momentum
    
    5. ATR (Average True Range):
       - Mede volatilidade do ativo
       - Usado para definir stop-loss dinâmico
       - ATR alto = maior volatilidade = maior risco
    
    6. VOLUME:
       - Confirma a força do movimento
       - Volume alto + alta de preço = tendência forte
       - Volume baixo + movimento = possível reversão
    
    7. ROC (Rate of Change):
       - Taxa de variação percentual do preço
       - Mede velocidade da mudança de preço
       - ROC positivo/crescente = momentum de alta
    """
    
    @staticmethod
    def calculate_all(df):
        """Calcula todos os indicadores técnicos"""
        data = df.copy()
        
        # === RETORNOS ===
        data['returns'] = data['Close'].pct_change()
        data['returns_5d'] = data['Close'].pct_change(5)
        data['returns_10d'] = data['Close'].pct_change(10)
        
        # === MÉDIAS MÓVEIS ===
        data['sma_9'] = data['Close'].rolling(window=9).mean()
        data['sma_21'] = data['Close'].rolling(window=21).mean()
        data['sma_50'] = data['Close'].rolling(window=50).mean()
        data['ema_9'] = data['Close'].ewm(span=9, adjust=False).mean()
        data['ema_21'] = data['Close'].ewm(span=21, adjust=False).mean()
        
        # Distância das médias (normalizado)
        data['dist_sma9'] = (data['Close'] - data['sma_9']) / data['Close']
        data['dist_sma21'] = (data['Close'] - data['sma_21']) / data['Close']
        
        # Cruzamento de médias
        data['ma_cross'] = np.where(data['sma_9'] > data['sma_21'], 1, -1)
        
        # === RSI ===
        data['rsi'] = TechnicalIndicators._calculate_rsi(data['Close'], 14)
        data['rsi_30'] = (data['rsi'] < 30).astype(int)  # Sobrevendido
        data['rsi_70'] = (data['rsi'] > 70).astype(int)  # Sobrecomprado
        
        # === BOLLINGER BANDS ===
        data['bb_middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_position'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # === MACD ===
        ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['macd'] = ema_12 - ema_26
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']
        data['macd_cross'] = np.where(data['macd'] > data['macd_signal'], 1, -1)
        
        # === VOLATILIDADE ===
        data['volatility'] = data['returns'].rolling(window=20).std() * np.sqrt(252)
        data['atr'] = TechnicalIndicators._calculate_atr(data, 14)
        data['atr_percent'] = data['atr'] / data['Close']
        
        # === VOLUME ===
        data['volume_sma'] = data['Volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_sma']
        data['volume_trend'] = data['Volume'].rolling(window=5).mean() / data['Volume'].rolling(window=20).mean()
        
        # === MOMENTUM ===
        data['momentum_10'] = data['Close'] - data['Close'].shift(10)
        data['roc'] = ((data['Close'] - data['Close'].shift(10)) / data['Close'].shift(10)) * 100
        
        # === PRICE ACTION ===
        data['high_low_ratio'] = (data['High'] - data['Low']) / data['Close']
        data['close_position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
        
        # Gap de abertura
        data['gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        
        return data
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        """Calcula RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_atr(df, period=14):
        """Calcula Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()


class PortfolioModel:
    """
    Modelo de Portfolio com múltiplas ações
    """
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.scaler = StandardScaler()
        self.models = {}  # Um modelo por ação
        
        # Features a serem usadas
        self.feature_cols = [
            'returns', 'returns_5d', 'rsi', 'bb_position', 'bb_width',
            'macd_hist', 'volatility', 'atr_percent', 'volume_ratio',
            'roc', 'dist_sma9', 'dist_sma21', 'ma_cross', 'macd_cross',
            'high_low_ratio', 'close_position', 'gap'
        ]
    
    def prepare_data(self, df, ticker, forward_periods=1, threshold=0.005):
        """
        Prepara dados para uma ação específica
        threshold: 0.5% mínimo para considerar sinal (ajustável)
        """
        data = TechnicalIndicators.calculate_all(df)
        
        # Target: retorno do próximo dia
        data['future_return'] = data['Close'].shift(-forward_periods) / data['Close'] - 1
        
        # Sinal: 1 (Compra), 0 (Neutro), -1 (Venda)
        data['signal'] = 0
        data.loc[data['future_return'] > threshold, 'signal'] = 1
        data.loc[data['future_return'] < -threshold, 'signal'] = -1
        
        # Remove NaN
        data = data.dropna()
        
        if len(data) > 0:
            X = data[self.feature_cols]
            y = data['signal']
            return X, y, data
        return None, None, None
    
    def train_models(self, data_dict, model_type='random_forest'):
        """
        Treina um modelo para cada ação
        data_dict: {ticker: DataFrame}
        """
        print("Treinando modelos...")
        
        for ticker in data_dict.keys():
            print(f"  Treinando {ticker}...", end=' ')
            
            X, y, _ = self.prepare_data(data_dict[ticker], ticker)
            
            if X is not None and len(X) > 100:
                # Split: 70% treino, 30% será usado no backtest
                split = int(len(X) * 0.7)
                X_train, y_train = X[:split], y[:split]
                
                # Normaliza
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                
                # Treina modelo
                if model_type == 'random_forest':
                    model = RandomForestClassifier(
                        n_estimators=100, 
                        max_depth=6,
                        min_samples_split=20,
                        random_state=42
                    )
                else:
                    model = LogisticRegression(max_iter=1000, random_state=42)
                
                model.fit(X_train_scaled, y_train)
                
                self.models[ticker] = {
                    'model': model,
                    'scaler': scaler,
                    'train_end_idx': split
                }
                print("✓")
            else:
                print("✗ (dados insuficientes)")
    
    def predict_signals(self, data_dict, date):
        """
        Gera sinais para todos os ativos em uma data específica
        Retorna: {ticker: (signal, probability, features)}
        """
        signals = {}
        
        for ticker, df in data_dict.items():
            if ticker not in self.models:
                continue
            
            X, y, data = self.prepare_data(df, ticker)
            
            if X is None or date not in data.index:
                continue
            
            # Pega features da data específica
            X_current = X.loc[[date]]
            
            # Normaliza e prediz
            scaler = self.models[ticker]['scaler']
            model = self.models[ticker]['model']
            
            X_scaled = scaler.transform(X_current)
            signal = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            
            # Pega volatilidade e retorno esperado
            volatility = data.loc[date, 'volatility']
            
            signals[ticker] = {
                'signal': signal,
                'probability': proba,
                'volatility': volatility
            }
        
        return signals
    
    def calculate_position_sizes(self, signals, available_capital):
        """
        Calcula tamanho das posições usando Kelly Criterion modificado
        """
        positions = {}
        
        # Filtra apenas sinais de compra
        buy_signals = {k: v for k, v in signals.items() if v['signal'] == 1}
        
        if not buy_signals:
            return positions
        
        # Calcula score de confiança (probabilidade * (1 - volatilidade))
        for ticker, data in buy_signals.items():
            # Probabilidade de compra (classe 1)
            prob_buy = data['probability'][2] if len(data['probability']) > 2 else data['probability'][1]
            
            # Ajusta por volatilidade (menor volatilidade = maior confiança)
            vol_penalty = min(data['volatility'], 0.5)  # Cap em 50%
            confidence = prob_buy * (1 - vol_penalty)
            
            buy_signals[ticker]['confidence'] = confidence
        
        # Normaliza confiança
        total_confidence = sum([v['confidence'] for v in buy_signals.values()])
        
        if total_confidence > 0:
            for ticker, data in buy_signals.items():
                weight = data['confidence'] / total_confidence
                # Limita posição individual a 20% do capital
                weight = min(weight, 0.20)
                positions[ticker] = available_capital * weight
        
        return positions
    
    def backtest(self, data_dict, start_date=None, end_date=None, 
                 transaction_cost=0.0003, max_positions=10):
        """
        Realiza backtest completo do portfolio
        
        transaction_cost: 0.03% por operação (compra + venda = 0.06%)
        max_positions: máximo de posições simultâneas
        """
        print("\n" + "="*70)
        print("INICIANDO BACKTEST")
        print("="*70)
        
        # Inicializa portfolio
        cash = self.initial_capital
        positions = {}  # {ticker: {shares, entry_price, entry_date}}
        portfolio_value = []
        trades_history = []
        
        # Determina datas de backtest
        all_dates = sorted(set.union(*[set(df.index) for df in data_dict.values()]))
        
        # Usa apenas período de teste (após treino)
        test_dates = all_dates
        for ticker in self.models.keys():
            train_end = self.models[ticker]['train_end_idx']
            X, _, data = self.prepare_data(data_dict[ticker], ticker)
            if X is not None:
                test_start_date = data.index[train_end]
                test_dates = [d for d in test_dates if d >= test_start_date]
                break
        
        if start_date:
            test_dates = [d for d in test_dates if d >= pd.Timestamp(start_date)]
        if end_date:
            test_dates = [d for d in test_dates if d <= pd.Timestamp(end_date)]
        
        print(f"Período: {test_dates[0].date()} a {test_dates[-1].date()}")
        print(f"Total de dias: {len(test_dates)}")
        print(f"Capital inicial: R$ {self.initial_capital:,.2f}")
        print(f"Custo transação: {transaction_cost*100:.2f}%")
        print(f"Máx posições: {max_positions}\n")
        
        # Itera pelos dias
        for i, date in enumerate(test_dates):
            if i % 100 == 0:
                print(f"Processando dia {i+1}/{len(test_dates)}...")
            
            # Atualiza valor das posições abertas
            current_portfolio_value = cash
            for ticker, pos in list(positions.items()):
                if date in data_dict[ticker].index:
                    current_price = data_dict[ticker].loc[date, 'Close']
                    pos['current_price'] = current_price
                    pos['current_value'] = pos['shares'] * current_price
                    pos['return'] = (current_price / pos['entry_price']) - 1
                    current_portfolio_value += pos['current_value']
            
            # Gera sinais
            signals = self.predict_signals(data_dict, date)
            
            # FECHA POSIÇÕES (sinais de venda ou neutros)
            for ticker in list(positions.keys()):
                should_close = False
                
                if ticker in signals:
                    if signals[ticker]['signal'] <= 0:  # Venda ou neutro
                        should_close = True
                    # Stop loss: fecha se perda > 5%
                    elif positions[ticker]['return'] < -0.05:
                        should_close = True
                    # Take profit: fecha se ganho > 10%
                    elif positions[ticker]['return'] > 0.10:
                        should_close = True
                
                if should_close and date in data_dict[ticker].index:
                    pos = positions[ticker]
                    exit_price = data_dict[ticker].loc[date, 'Close']
                    proceeds = pos['shares'] * exit_price
                    cost = proceeds * transaction_cost
                    cash += proceeds - cost
                    
                    ret = (exit_price / pos['entry_price']) - 1
                    
                    trades_history.append({
                        'ticker': ticker,
                        'entry_date': pos['entry_date'],
                        'exit_date': date,
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'shares': pos['shares'],
                        'return': ret,
                        'profit': (exit_price - pos['entry_price']) * pos['shares'] - cost
                    })
                    
                    del positions[ticker]
            
            # ABRE NOVAS POSIÇÕES
            if len(positions) < max_positions:
                available_cash = cash * 0.95  # Usa no máximo 95% do caixa
                new_positions = self.calculate_position_sizes(signals, available_cash)
                
                for ticker, capital in new_positions.items():
                    if ticker not in positions and date in data_dict[ticker].index:
                        entry_price = data_dict[ticker].loc[date, 'Close']
                        cost = capital * transaction_cost
                        shares = (capital - cost) / entry_price
                        
                        if shares > 0:
                            positions[ticker] = {
                                'shares': shares,
                                'entry_price': entry_price,
                                'entry_date': date,
                                'current_price': entry_price,
                                'current_value': capital - cost,
                                'return': 0
                            }
                            cash -= capital
                        
                        if len(positions) >= max_positions:
                            break
            
            # Registra valor do portfolio
            portfolio_value.append({
                'date': date,
                'total_value': current_portfolio_value,
                'cash': cash,
                'n_positions': len(positions)
            })
        
        # Fecha todas as posições no final
        final_date = test_dates[-1]
        for ticker, pos in positions.items():
            if final_date in data_dict[ticker].index:
                exit_price = data_dict[ticker].loc[final_date, 'Close']
                proceeds = pos['shares'] * exit_price
                cost = proceeds * transaction_cost
                cash += proceeds - cost
        
        # Converte para DataFrames
        portfolio_df = pd.DataFrame(portfolio_value)
        trades_df = pd.DataFrame(trades_history)
        
        # Calcula métricas
        self._print_backtest_results(portfolio_df, trades_df)
        
        return portfolio_df, trades_df
    
    def _print_backtest_results(self, portfolio_df, trades_df):
        """Imprime resultados do backtest"""
        
        final_value = portfolio_df['total_value'].iloc[-1]
        total_return = (final_value / self.initial_capital) - 1
        
        # Retornos diários
        portfolio_df['daily_return'] = portfolio_df['total_value'].pct_change()
        
        # Métricas
        days = len(portfolio_df)
        years = days / 252
        cagr = (final_value / self.initial_capital) ** (1/years) - 1 if years > 0 else 0
        
        volatility = portfolio_df['daily_return'].std() * np.sqrt(252)
        sharpe = (portfolio_df['daily_return'].mean() * 252) / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + portfolio_df['daily_return']).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trades
        if len(trades_df) > 0:
            win_rate = (trades_df['return'] > 0).mean()
            avg_win = trades_df[trades_df['return'] > 0]['return'].mean()
            avg_loss = trades_df[trades_df['return'] < 0]['return'].mean()
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        print("\n" + "="*70)
        print("RESULTADOS DO BACKTEST")
        print("="*70)
        print(f"\nCAPITAL:")
        print(f"  Inicial: R$ {self.initial_capital:,.2f}")
        print(f"  Final: R$ {final_value:,.2f}")
        print(f"  Lucro: R$ {final_value - self.initial_capital:,.2f}")
        print(f"\nRETORNO:")
        print(f"  Total: {total_return*100:.2f}%")
        print(f"  CAGR: {cagr*100:.2f}% ao ano")
        print(f"\nRISCO:")
        print(f"  Volatilidade: {volatility*100:.2f}%")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Max Drawdown: {max_drawdown*100:.2f}%")
        print(f"\nTRADES:")
        print(f"  Total: {len(trades_df)}")
        print(f"  Win Rate: {win_rate*100:.1f}%")
        print(f"  Retorno médio (ganho): {avg_win*100:.2f}%")
        print(f"  Retorno médio (perda): {avg_loss*100:.2f}%")
        print(f"  Profit Factor: {profit_factor:.2f}")
        print("="*70)


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║        SISTEMA DE PORTFOLIO TRADING QUANTITATIVO - B3           ║
    ║                    Backtest 5 Anos                               ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Simula dados para demonstração
    # EM PRODUÇÃO: substituir por dados reais do yfinance
    np.random.seed(42)
    
    dates = pd.date_range(start='2019-01-01', end='2024-12-31', freq='B')
    
    # Simula 10 ações com características diferentes
    data_dict = {}
    tickers_sample = TOP30_B3[:10]
    
    print("Gerando dados simulados...")
    for ticker in tickers_sample:
        # Cada ação tem drift e volatilidade diferentes
        drift = np.random.uniform(-0.0001, 0.0003)
        vol = np.random.uniform(0.015, 0.030)
        
        returns = np.random.normal(drift, vol, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.randn(len(dates)) * 0.005),
            'High': prices * (1 + np.abs(np.random.randn(len(dates))) * 0.01),
            'Low': prices * (1 - np.abs(np.random.randn(len(dates))) * 0.01),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        data_dict[ticker] = df
    
    print(f"✓ Dados gerados para {len(data_dict)} ações\n")
    
    # Cria e treina modelo
    portfolio = PortfolioModel(initial_capital=100000)
    portfolio.train_models(data_dict, model_type='random_forest')
    
    # Executa backtest
    portfolio_history, trades = portfolio.backtest(
        data_dict,
        transaction_cost=0.0003,
        max_positions=5
    )
    
    print("\n✓ Backtest concluído!")
    print("\nPróximos passos:")
    print("1. Substituir dados simulados por dados reais (yfinance)")
    print("2. Ajustar hiperparâmetros (threshold, max_positions, etc)")
    print("3. Adicionar mais features ou modelos")
    print("4. Implementar análise de sensibilidade")