# 🌫️ Qualidade do Ar nas Grandes Cidades Brasileiras (2015–2024)

Projeto de análise e visualização de dados com **Python, Pandas, Matplotlib, Seaborn, Plotly, Streamlit, SQLAlchemy/SQLite e GitHub**.

| | |
|---|---|
| **Aluno** | Gabriel Hierro Amorim dos Santos |
| **Professor** | Alexandre Neves Louzada |
| **Disciplina** | Linguagem de Programação — Análise e Visualização de Dados com Python |
| **Avaliação** | G1 · Projeto Tema 08 |
| **Data** | Outubro de 2026 |

## 🔗 Links do projeto

| Item | Link |
|---|---|
| 💻 Repositório (GitHub) | https://github.com/gabrielhierro/projeto-qualidade-ar |
| 🌐 Página do projeto (GitHub Pages) | https://gabrielhierro.github.io/projeto-qualidade-ar/ |
| 🚀 Dashboard (Streamlit Community Cloud) | https://projeto-qualidade-ar.streamlit.app/ |
| 📓 Notebook de análise | [`notebooks/analise_qualidade_ar.ipynb`](notebooks/analise_qualidade_ar.ipynb) |
| 🧩 Código do dashboard | [`app.py`](app.py) |
| 🗂️ Base de dados | [`dados/simulacao_qualidade_ar_brasil.csv`](dados/simulacao_qualidade_ar_brasil.csv) |

## 📌 Contexto e objetivo

A qualidade do ar influencia diretamente a saúde pública, a qualidade de vida, o meio ambiente, a mobilidade urbana e a sustentabilidade das cidades. Trânsito intenso, atividades industriais, queimadas, baixa circulação de ar e crescimento urbano desordenado elevam a concentração de poluentes nos grandes centros.

Este projeto analisa **4.440 medições mensais simuladas de 37 cidades brasileiras (20 estados, 5 regiões), de janeiro de 2015 a dezembro de 2024**, para identificar cidades, períodos e poluentes que exigem maior atenção ambiental.

**Decisão que a análise apoia:** definir onde e em qual poluente concentrar ações de monitoramento e controle de emissões, e avaliar se há períodos do ano que justifiquem medidas sazonais.

### Perguntas orientadoras

1. Quais cidades apresentam pior qualidade do ar?
2. Existem períodos mais críticos?
3. Quais poluentes são mais frequentes?
4. Existe relação entre clima e poluição?
5. Há melhora ou piora da qualidade do ar ao longo do tempo?
6. Quais regiões apresentam maior concentração de poluentes?
7. Quais cidades exigem maior atenção ambiental?

## 🗃️ Base de dados

Arquivo: `dados/simulacao_qualidade_ar_brasil.csv` (dataset **simulado**, fornecido para a disciplina). Granularidade mensal: 1 registro por cidade e mês (37 cidades × 120 meses = 4.440 registros).

| Coluna | Descrição |
|---|---|
| `ano`, `mes`, `data` | Período de referência |
| `regiao`, `uf`, `cidade` | Localização monitorada |
| `pm25`, `pm10`, `no2`, `co`, `o3` | Concentração média mensal dos poluentes (unidade não informada na base) |
| `temperatura_media` | Temperatura média do mês (°C) |
| `umidade` | Umidade relativa média do mês (%) |
| `indice_qualidade_ar` | Índice geral de qualidade do ar (quanto maior, pior) |
| `nivel_qualidade` | Boa (< 50), Moderada (50–80) ou Ruim (≥ 80); nenhum registro atingiu “Péssima” |

## 🧹 Tratamento e engenharia de atributos

- Verificação de nulos e duplicados (nenhum encontrado), padronização de textos e conversão de tipos.
- Validações de consistência: período, localização (cada cidade em uma única UF e região) e coerência entre índice e nível de qualidade.
- Análise de valores extremos pelo critério IQR (menos de 1% dos registros; mantidos por serem plausíveis).
- Atributos criados: nome do mês, trimestre, estação do ano, indicador de registro crítico, contribuição de cada poluente ao índice, poluente predominante e faixas de temperatura e umidade.
- Persistência em **SQLite** com **SQLAlchemy**, em duas tabelas relacionadas (`cidades` e `medicoes`).

## 📊 KPIs

| KPI | Resultado |
|---|---|
| Índice médio de qualidade do ar | **67,74** |
| Cidade mais poluída | **Florianópolis (SC)** — índice médio 70,29 |
| Poluente predominante | **PM2.5** — 46,6% da composição do índice |
| Percentual de períodos críticos (Ruim/Péssima) | **11,8%** (524 de 4.440 registros cidade-mês) |
| Região mais afetada | **Sul** — 67,92 (empate técnico com o Sudeste) |
| Média de PM2.5 | **35,06** |

## 🔎 Principais resultados

- **Cidades:** Florianópolis, Petrópolis, Joinville, São Paulo e Campinas têm os maiores índices médios; Florianópolis e Joinville também lideram em registros críticos (18,3% e 19,2%). A diferença entre a melhor e a pior cidade é de apenas 3,9 pontos (ANOVA, p = 0,59).
- **Poluentes:** o PM2.5 é o maior contribuinte do índice em 93,5% dos registros (correlação r = 0,80 com o índice).
- **Períodos:** março, agosto e junho concentram mais registros críticos, mas não há sazonalidade estatisticamente significativa (p = 0,38).
- **Tendência:** estabilidade ao longo de 2015–2024 (−0,03 ponto por ano, p = 0,67).
- **Clima × poluição:** sem relação relevante (índice × temperatura r = 0,01; índice × umidade r = −0,03).
- **Regiões:** nenhuma região se destaca de forma consistente (diferença entre extremos de 0,5 ponto).

> ⚠️ **Limitação importante:** a base é simulada. Os resultados descrevem o comportamento dos dados fornecidos e não representam a situação real da qualidade do ar de cada cidade.

## 🖥️ Dashboard Streamlit

O dashboard reúne título e descrição do problema, **KPIs**, **filtros interativos** (ano, mês, região, estado, cidade e nível de qualidade), **gráficos** (linha temporal, barras por cidade e por poluente, heatmap mensal e dispersão temperatura × poluição), **tabela dinâmica**, **interpretação textual** e **conclusão executiva**.

👉 https://projeto-qualidade-ar.streamlit.app/

## 🗂️ Estrutura do repositório

```
projeto-qualidade-ar/
├── app.py                  # dashboard Streamlit
├── requirements.txt        # dependências
├── README.md               # este arquivo
├── index.html              # página do projeto (GitHub Pages)
├── dados/                  # base de dados (CSV)
├── database/               # banco SQLite (qualidade_ar.db)
├── notebooks/
│   └── analise_qualidade_ar.ipynb
└── imagens/                # gráficos exportados do notebook
```

## ▶️ Como executar localmente

```bash
# 1. clonar o repositório
git clone https://github.com/gabrielhierro/projeto-qualidade-ar.git
cd projeto-qualidade-ar

# 2. (opcional) criar ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. instalar dependências
pip install -r requirements.txt

# 4. executar o dashboard
streamlit run app.py
```

Para reproduzir a análise, abra `notebooks/analise_qualidade_ar.ipynb` no Jupyter ou no Google Colab.

## 🛠️ Tecnologias

Python · Pandas · NumPy · SciPy · Matplotlib · Seaborn · Plotly · Streamlit · SQLAlchemy · SQLite · GitHub · GitHub Pages · Streamlit Community Cloud

---

Projeto desenvolvido para fins educacionais — Análise e Visualização de Dados com Python.
