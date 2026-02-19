# Janus: Dual Strategy – Relatório Final Desafio Quant-AI Itaú 2025

![Janus Robot](INSERIR_LINK_DA_IMAGEM_DO_ROBO_AQUI)

## 🤖 Sobre o Janus
O **Janus** é um robô de investimentos focado em ações da B3 (Universo IBrA). O nome é uma homenagem ao deus romano das transições, Jano, que possui duas faces. Essa dualidade reflete a arquitetura do modelo: uma "face" observa a tendência macro (passado recente), enquanto a outra aguarda o momento tático de entrada (recuo de curto prazo).

O projeto foi desenvolvido como um modelo quantitativo adaptativo que busca superar as limitações de estratégias monométricas através do conceito de **Regime Switching**.

---

## 📈 Tese de Investimento
A estratégia, denominada **Dual Strategy**, aloca capital dinamicamente baseando-se em dois regimes complementares:

* **Trend Following (TEND):** Busca capturar grandes movimentos e crescimento exponencial.
* **Reversão à Média (REV):** Otimiza o ponto de entrada através de recuos temporários em cenários de sobrevenda ou sobrecompra.

### Fluxograma da Lógica
![Fluxograma da Lógica Janus](INSERIR_LINK_DO_FLUXOGRAMA_AQUI)


---

## 🛠️ Funcionamento Técnico
O modelo utiliza um **Filtro de Tendência Macro** (baseado na porcentagem de ativos acima de suas médias de longo prazo) para definir o regime de mercado: **Bull, Bear ou Neutro**.

* **Seleção de Ativos:** Filtros estritos excluem papéis com preço abaixo de R$ 5,00 e com alta volatilidade relativa (ATR/Preço > 5%)[cite: 18].
* **Sistema de Scoring:** A decisão de compra ocorre quando o ativo excede um Score Mínimo baseado em indicadores como ADX, PDI/MDI (para tendência) e Keltner Channels/RSI (para reversão).
* **Frequência:** Diária, utilizando o benchmark IBOV.

---

## 🛡️ Gestão de Risco
[cite_start]O gerenciamento de risco é calibrado individualmente por estratégia usando o **ATR (Average True Range)**[cite: 13]:

| Estratégia | Risco Máximo (Capital Total) | Regra de Saída |
| :--- | :--- | :--- |
| **Tendência (TEND)** | 1.5% | [cite_start]Trailing Stop Móvel (2.5x ATR) [cite: 13, 16] |
| **Reversão (REV)** | 0.3% | [cite_start]Alvo Fixo (3x Risco) ou Keltner Superior [cite: 13, 15] |

---

## 📊 Resultados e Performance
[cite_start]Os testes abrangeram um período de **25.8 anos**, comprovando a capacidade de gerar **Alpha** com risco controlado[cite: 27, 28].

### Métricas Consolidadas (2000 - 2025)
* [cite_start]**Retorno Anual Médio (CAGR):** 12.68% [cite: 19]
* [cite_start]**Retorno Total:** 2066.63% [cite: 19]
* [cite_start]**Máximo Drawdown:** -38.74% (significativamente menor que o Ibovespa histórico) [cite: 19, 28]
* [cite_start]**Profit Factor:** 1.49x [cite: 19, 21]
* [cite_start]**Ratio Lucro/Prejuízo:** 2.86x [cite: 19, 30]

![Gráfico Evolução do Capital](INSERIR_LINK_DO_GRAFICO_DE_CAPITAL_AQUI)
![Gráfico de Drawdown](INSERIR_LINK_DO_GRAFICO_DE_DRAWDOWN_AQUI)

---

## 🧠 Desenvolvimento e IA
[cite_start]O projeto utilizou ferramentas de IA Generativa (**ChatGPT, Claude e Gemini**) como apoio técnico[cite: 37, 38]:
* [cite_start]Depuração e otimização de código Python[cite: 41].
* [cite_start]Calibragem de parâmetros (scores e métricas de risco) e análise de sensibilidade no backtest[cite: 43].
* [cite_start]Criação de identidade visual e auxílio na estruturação do relatório[cite: 40, 43].

> [cite_start]**Nota:** Todas as respostas geradas por IA foram submetidas à análise crítica e supervisão humana para corrigir inconsistências lógicas ou erros conceituais[cite: 39, 46, 47].

---

## 🚀 Desafios Futuros
* [cite_start]Otimização da performance em regimes de mercado lateral/neutro, onde o Profit Factor tende a cair[cite: 31].
* [cite_start]Implementação de filtros por setores para aumentar a resiliência e diversificação da carteira[cite: 31].

---
[cite_start]*Este projeto foi desenvolvido para o Desafio Quant-AI Itaú 2025.* [cite: 1]
