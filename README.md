# Janus: Dual Strategy – Relatório Final Desafio Quant-AI Itaú 2025

![Janus Robot](INSERIR_LINK_DA_IMAGEM_DO_ROBO_AQUI)
## 🤖 Sobre o Janus
[cite_start] O **Janus** é um robô de investimentos focado em ações da B3 (Universo IBrA)[cite: 5, 6]. [cite_start]O nome é uma homenagem ao deus romano das transições, Jano, que possui duas faces[cite: 3]. [cite_start]Essa dualidade reflete a arquitetura do modelo: uma "face" observa a tendência macro (passado recente), enquanto a outra aguarda o momento tático de entrada (recuo de curto prazo)[cite: 3].

[cite_start]O projeto foi desenvolvido como um modelo quantitativo adaptativo que busca superar as limitações de estratégias monométricas através do conceito de **Regime Switching**[cite: 5, 6].

---

## 📈 Tese de Investimento
[cite_start]A estratégia, denominada **Dual Strategy**, aloca capital dinamicamente baseando-se em dois regimes complementares[cite: 6]:

* [cite_start]**Trend Following (TEND):** Busca capturar grandes movimentos e crescimento exponencial[cite: 8].
* [cite_start]**Reversão à Média (REV):** Otimiza o ponto de entrada através de recuos temporários em cenários de sobrevenda ou sobrecompra[cite: 4, 8].

### Fluxograma da Lógica
[cite_start][cite: 33]

---

## 🛠️ Funcionamento Técnico
[cite_start]O modelo utiliza um **Filtro de Tendência Macro** para definir o regime de mercado: **Bull, Bear ou Neutro**[cite: 7].

* [cite_start]**Seleção de Ativos:** Filtros estritos excluem papéis abaixo de R$ 5,00 e com volatilidade excessiva (ATR/Preço > 5%)[cite: 18].
* [cite_start]**Sistema de Scoring:** A decisão de compra ocorre quando o ativo excede um Score Mínimo baseado em indicadores como ADX, PDI/MDI (para tendência) e Keltner Channels/RSI (para reversão)[cite: 9, 10, 11, 12].
* [cite_start]**Frequência:** Diária, utilizando o benchmark IBOV[cite: 7, 8].

---

## 🛡️ Gestão de Risco
[cite_start]O gerenciamento de risco é calibrado individualmente por estratégia usando o **ATR (Average True Range)**[cite: 13]:

| Estratégia | Risco Máximo (Capital Total) | Regra de Saída |
| :--- | :--- | :--- |
| **Tendência (TEND)** | [cite_start]1.5% [cite: 13] | [cite_start]Trailing Stop Móvel (2.5x ATR) [cite: 16] |
| **Reversão (REV)** | [cite_start]0.3% [cite: 13] | [cite_start]Alvo Fixo (3x Risco) ou Keltner Superior [cite: 15] |

---

## 📊 Resultados e Performance
[cite_start]Os testes abrangeram um período de **25.8 anos**, comprovando a capacidade de gerar **Alpha** com risco controlado[cite: 20, 27].

### Métricas Consolidadas (2000 - 2025)
* [cite_start]**Retorno Anual Médio (CAGR):** 12.68% [cite: 19]
* [cite_start]**Retorno Total:** 2066.63% [cite: 19]
* [cite_start]**Máximo Drawdown:** -38.74% (significativamente menor que o Ibovespa histórico) [cite: 19, 20]
* [cite_start]**Profit Factor:** 1.49x [cite: 19]
* [cite_start]**Ratio Lucro/Prejuízo:** 2.86x [cite: 19]

[cite_start][cite: 34]

---

## 🧠 Desenvolvimento e IA
[cite_start]O projeto utilizou ferramentas de IA Generativa (**ChatGPT, Claude e Gemini**) como apoio técnico para[cite: 37, 38]:
* [cite_start]Depuração e otimização de código Python[cite: 41].
* [cite_start]Calibragem de parâmetros e análise de sensibilidade no backtest[cite: 42, 43].
* [cite_start]Criação de identidade visual e relatórios[cite: 43].

> [cite_start]**Nota:** Todas as sugestões da IA foram submetidas a validação manual e supervisão humana para corrigir inconsistências lógicas[cite: 46, 47].

---

## 🚀 Próximos Passos
* [cite_start]Otimização da performance em regimes de mercado lateral/neutro[cite: 31].
* [cite_start]Implementação de filtros por setores para aumentar a diversificação[cite: 31].

---
[cite_start]*Este projeto foi desenvolvido para o Desafio Quant-AI Itaú 2025.* [cite: 1]
