# -*- coding: utf-8 -*-
"""
Dashboard: Qualidade do Ar nas Grandes Cidades Brasileiras (2015-2024)
Projeto G1 - Análise e Visualização de Dados com Python (Tema 08)

Disciplina: Linguagem de Programação - Análise e Visualização de Dados com Python
Professor : Alexandre Neves Louzada
Aluno     : Gabriel Hierro Amorim dos Santos
"""
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from scipy import stats

# ---------------------------------------------------------------------------
# Identificação do projeto
# ---------------------------------------------------------------------------
DISCIPLINA = "Linguagem de Programação – Análise e Visualização de Dados com Python"
PROFESSOR = "Alexandre Neves Louzada"
ALUNO = "Gabriel Hierro Amorim dos Santos"
TITULO = "Qualidade do Ar nas Grandes Cidades Brasileiras (2015–2024)"

st.set_page_config(page_title="Qualidade do Ar no Brasil", page_icon="🌫️", layout="wide")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
CAMINHO_DADOS = Path(__file__).parent / "dados" / "simulacao_qualidade_ar_brasil.csv"

POLUENTES = ["pm25", "pm10", "no2", "co", "o3"]
NOMES = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO2", "co": "CO", "o3": "O3"}
INDICADORES = {"Índice de qualidade do ar": "indice_qualidade_ar", "PM2.5": "pm25", "PM10": "pm10",
               "NO2": "no2", "CO": "co", "O3": "o3"}
COLUNAS_OBRIGATORIAS = ["ano", "mes", "data", "regiao", "uf", "cidade", "pm25", "pm10", "no2", "co", "o3",
                        "temperatura_media", "umidade", "indice_qualidade_ar", "nivel_qualidade"]

MESES_PT = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
            7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}
ESTACOES = {12: "Verão", 1: "Verão", 2: "Verão", 3: "Outono", 4: "Outono", 5: "Outono",
            6: "Inverno", 7: "Inverno", 8: "Inverno", 9: "Primavera", 10: "Primavera", 11: "Primavera"}
ORDEM_ESTACOES = ["Verão", "Outono", "Inverno", "Primavera"]
NIVEIS = ["Boa", "Moderada", "Ruim", "Péssima"]
ORDEM_REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]

CORES_REGIAO = {"Norte": "#2E8B57", "Nordeste": "#E69F00", "Centro-Oeste": "#8C564B",
                "Sudeste": "#0072B2", "Sul": "#9467BD"}
CORES_NIVEL = {"Boa": "#2E9E5B", "Moderada": "#F2B134", "Ruim": "#D64541", "Péssima": "#7B1E1E"}
COR_DESTAQUE, COR_APOIO, COR_NEUTRA = "#C0392B", "#2E86AB", "#9AA5B1"

COORDENADAS = {
    "Manaus": (-3.119, -60.022), "Belém": (-1.456, -48.502), "Santarém": (-2.443, -54.708),
    "Porto Velho": (-8.761, -63.900), "Palmas": (-10.184, -48.333), "Salvador": (-12.971, -38.501),
    "Feira de Santana": (-12.267, -38.967), "Recife": (-8.054, -34.881),
    "Jaboatão dos Guararapes": (-8.113, -35.015), "Fortaleza": (-3.732, -38.527),
    "Juazeiro do Norte": (-7.213, -39.315), "São Luís": (-2.530, -44.303), "João Pessoa": (-7.115, -34.863),
    "Brasília": (-15.794, -47.882), "Goiânia": (-16.686, -49.264), "Aparecida de Goiânia": (-16.823, -49.247),
    "Cuiabá": (-15.601, -56.097), "Campo Grande": (-20.469, -54.620), "São Paulo": (-23.550, -46.633),
    "Campinas": (-22.906, -47.061), "Ribeirão Preto": (-21.178, -47.810), "Rio de Janeiro": (-22.907, -43.173),
    "Niterói": (-22.883, -43.104), "Nova Iguaçu": (-22.759, -43.451), "Petrópolis": (-22.505, -43.179),
    "Belo Horizonte": (-19.917, -43.934), "Juiz de Fora": (-21.764, -43.350), "Uberlândia": (-18.919, -48.277),
    "Vitória": (-20.315, -40.312), "Vila Velha": (-20.329, -40.292), "Serra": (-20.121, -40.307),
    "Curitiba": (-25.428, -49.273), "Londrina": (-23.310, -51.163), "Porto Alegre": (-30.035, -51.218),
    "Caxias do Sul": (-29.168, -51.179), "Florianópolis": (-27.595, -48.548), "Joinville": (-26.304, -48.846),
}


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------
def fmt(valor, casas=1):
    """Formata número no padrão brasileiro (vírgula decimal e ponto de milhar)."""
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def forca_correlacao(r):
    r = abs(r)
    if r < 0.1:
        return "desprezível"
    if r < 0.3:
        return "fraca"
    if r < 0.5:
        return "moderada"
    return "forte"


def anova_p(d, coluna, alvo="indice_qualidade_ar"):
    """p-valor da ANOVA do índice entre os grupos de `coluna` (None se não aplicável)."""
    grupos = [g[alvo].values for _, g in d.groupby(coluna) if len(g) >= 2]
    if len(grupos) < 2:
        return None
    try:
        return float(stats.f_oneway(*grupos).pvalue)
    except Exception:
        return None


def preparar(bruto):
    """Limpeza e engenharia de atributos (mesmas etapas do notebook de análise)."""
    ausentes = [c for c in COLUNAS_OBRIGATORIAS if c not in [x.strip().lower() for x in bruto.columns]]
    if ausentes:
        raise ValueError("Colunas ausentes no arquivo: " + ", ".join(ausentes))

    df = bruto.copy()
    df.columns = df.columns.str.strip().str.lower()
    for col in ["regiao", "uf", "cidade", "nivel_qualidade"]:
        df[col] = df[col].astype(str).str.strip()
    df["uf"] = df["uf"].str.upper()
    df["data"] = pd.to_datetime(df["data"])
    df[["ano", "mes"]] = df[["ano", "mes"]].astype(int)
    df = df.dropna(subset=COLUNAS_OBRIGATORIAS).drop_duplicates().reset_index(drop=True)

    # Calendário e registro crítico
    df["mes_nome"] = df["mes"].map(MESES_PT)
    df["estacao"] = df["mes"].map(ESTACOES)
    df["critico"] = df["nivel_qualidade"].isin(["Ruim", "Péssima"])
    df["critico_pct"] = df["critico"].astype(float) * 100

    # Composição do índice (regressão linear do índice sobre os poluentes)
    X = np.column_stack([np.ones(len(df))] + [df[p] for p in POLUENTES])
    coef, *_ = np.linalg.lstsq(X, df["indice_qualidade_ar"], rcond=None)
    pesos = pd.Series(np.round(coef[1:], 2), index=POLUENTES)
    for p in POLUENTES:
        df[f"contrib_{p}"] = df[p] * pesos[p]
    colunas_contrib = [f"contrib_{p}" for p in POLUENTES]
    df["poluente_predominante"] = (df[colunas_contrib].idxmax(axis=1)
                                   .str.replace("contrib_", "", regex=False).map(NOMES))
    return df


@st.cache_data(show_spinner="Carregando dados...")
def carregar(conteudo):
    if conteudo is not None:
        bruto = pd.read_csv(io.BytesIO(conteudo), encoding="utf-8-sig")
    else:
        bruto = pd.read_csv(CAMINHO_DADOS, encoding="utf-8-sig")
    return preparar(bruto)


def decompor(serie):
    """Decomposição clássica aditiva (tendência 2x12, sazonalidade e resíduo)."""
    tendencia = serie.rolling(12).mean().rolling(2).mean().shift(-6)
    sem_tendencia = serie - tendencia
    indice = sem_tendencia.groupby(sem_tendencia.index.month).mean()
    indice = indice - indice.mean()
    sazonal = pd.Series(serie.index.month.map(indice).values, index=serie.index)
    residuo = serie - tendencia - sazonal
    ok = residuo.notna()
    f_saz = max(0.0, 1 - residuo[ok].var() / (sazonal[ok] + residuo[ok]).var())
    f_ten = max(0.0, 1 - residuo[ok].var() / (tendencia[ok] + residuo[ok]).var())
    return tendencia, sazonal, residuo, f_saz, f_ten


def figura_padrao(fig, altura=420, titulo=None):
    fig.update_layout(template="plotly_white", height=altura, margin=dict(l=10, r=10, t=60, b=10),
                      title=dict(text=titulo, x=0.0) if titulo else None)
    return fig


# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------
st.title(f"🌫️ {TITULO}")
st.markdown(
    f"**Disciplina:** {DISCIPLINA}  \n**Professor:** {PROFESSOR}  \n**Aluno:** {ALUNO}"
)

with st.container(border=True):
    st.subheader("Descrição do problema")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Pergunta**  \nQuais cidades e regiões apresentam pior qualidade do ar, em quais períodos "
                "e com qual poluente predominante? Existe relação com o clima ou tendência ao longo do tempo?")
    c2.markdown("**Por que importa**  \nA poluição atmosférica afeta a saúde pública, a mobilidade e a "
                "sustentabilidade urbana. Recursos de monitoramento são limitados e precisam ser priorizados.")
    c3.markdown("**Decisão apoiada**  \nDefinir onde e em qual poluente concentrar ações de monitoramento e "
                "controle de emissões, e avaliar se há períodos que justifiquem medidas sazonais.")
    st.caption("Base: dataset **simulado** com 37 cidades brasileiras, medições mensais de 2015 a 2024.")

# ---------------------------------------------------------------------------
# Dados e filtros (barra lateral)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌫️ Qualidade do Ar")
    st.caption(f"{DISCIPLINA}  \nProf. {PROFESSOR}  \n{ALUNO}")
    with st.expander("Fonte de dados"):
        arquivo = st.file_uploader("Carregar outro CSV (mesma estrutura)", type="csv")
        st.caption("Sem upload, é usada a base `dados/simulacao_qualidade_ar_brasil.csv`.")

try:
    df = carregar(arquivo.getvalue() if arquivo is not None else None)
except Exception as erro:
    st.error(f"Não foi possível carregar os dados: {erro}")
    st.stop()

with st.sidebar:
    st.header("Filtros")
    anos_opc = sorted(df["ano"].unique())
    anos = st.multiselect("Ano", anos_opc, default=anos_opc)
    meses = st.multiselect("Mês", list(range(1, 13)), default=list(range(1, 13)), format_func=lambda m: MESES_PT[m])
    regioes_opc = [r for r in ORDEM_REGIOES if r in set(df["regiao"])] + \
                  sorted(set(df["regiao"]) - set(ORDEM_REGIOES))
    regioes = st.multiselect("Região", regioes_opc, default=regioes_opc)
    ufs_opc = sorted(df.loc[df["regiao"].isin(regioes), "uf"].unique())
    ufs = st.multiselect("Estado (UF)", ufs_opc, default=ufs_opc)
    cidades_opc = sorted(df.loc[df["regiao"].isin(regioes) & df["uf"].isin(ufs), "cidade"].unique())
    cidades = st.multiselect("Cidade", cidades_opc, default=cidades_opc)
    niveis_opc = [n for n in NIVEIS if n in set(df["nivel_qualidade"])]
    niveis = st.multiselect("Nível de qualidade", niveis_opc, default=niveis_opc)
    st.caption("Os filtros de estado e cidade se ajustam à região escolhida. "
               "Todos os KPIs e gráficos refletem a seleção atual.")

mascara = (df["ano"].isin(anos) & df["mes"].isin(meses) & df["regiao"].isin(regioes) & df["uf"].isin(ufs)
           & df["cidade"].isin(cidades) & df["nivel_qualidade"].isin(niveis))
d = df[mascara]

if d.empty:
    st.warning("Nenhum registro corresponde aos filtros selecionados. Ajuste os filtros na barra lateral.")
    st.stop()

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
st.header("Indicadores-chave (KPIs)")

media_cidade = d.groupby("cidade")["indice_qualidade_ar"].mean()
media_regiao = d.groupby("regiao")["indice_qualidade_ar"].mean()
contrib_cols = [f"contrib_{p}" for p in POLUENTES]
contrib = d[contrib_cols].mean()
contrib.index = [NOMES[c.replace("contrib_", "")] for c in contrib.index]
participacao = 100 * contrib / contrib.sum()

cidade_top = media_cidade.idxmax()
regiao_top = media_regiao.idxmax()
poluente_top = participacao.idxmax()
uf_cidade = d.loc[d["cidade"] == cidade_top, "uf"].iloc[0]

iqa_medio = d["indice_qualidade_ar"].mean()
pct_critico = d["critico_pct"].mean()
pm25_medio = d["pm25"].mean()
iqa_base, crit_base, pm25_base = df["indice_qualidade_ar"].mean(), df["critico_pct"].mean(), df["pm25"].mean()

linha1 = st.columns(3)
linha2 = st.columns(3)
with linha1[0].container(border=True):
    st.metric("Índice médio de qualidade do ar", fmt(iqa_medio, 2),
              delta=f"{iqa_medio - iqa_base:+.2f} vs. base completa".replace(".", ","),
              delta_color="inverse", help="Média do índice (maior = pior). Faixas: Boa < 50, Moderada 50–80, Ruim ≥ 80.")
with linha1[1].container(border=True):
    st.metric("Cidade mais poluída", f"{cidade_top} ({uf_cidade})",
              delta=f"índice médio {fmt(media_cidade.max(), 2)}", delta_color="off",
              help="Cidade com o maior índice médio de qualidade do ar na seleção atual.")
with linha1[2].container(border=True):
    st.metric("Poluente predominante", poluente_top,
              delta=f"{fmt(participacao[poluente_top])}% da composição do índice", delta_color="off",
              help="Poluente com maior contribuição média (peso × concentração) para o índice.")
with linha2[0].container(border=True):
    st.metric("Percentual de períodos críticos", f"{fmt(pct_critico)}%",
              delta=f"{pct_critico - crit_base:+.1f} p.p. vs. base completa".replace(".", ","),
              delta_color="inverse",
              help="Percentual de registros cidade-mês com nível Ruim ou Péssima (a base é mensal).")
with linha2[1].container(border=True):
    st.metric("Região mais afetada", regiao_top, delta=f"índice médio {fmt(media_regiao.max(), 2)}",
              delta_color="off", help="Região com o maior índice médio na seleção atual.")
with linha2[2].container(border=True):
    st.metric("Média de PM2.5", fmt(pm25_medio, 2),
              delta=f"{pm25_medio - pm25_base:+.2f} vs. base completa".replace(".", ","),
              delta_color="inverse", help="Concentração média de PM2.5 (unidade da base).")

st.caption(f"{fmt(len(d), 0)} registros cidade-mês selecionados de {fmt(len(df), 0)} "
           f"({d['cidade'].nunique()} cidades, {d['ano'].nunique()} anos).")

# ---------------------------------------------------------------------------
# Análises
# ---------------------------------------------------------------------------
st.header("Análises")
abas = st.tabs(["📈 Evolução temporal", "🏙️ Cidades e regiões", "🧪 Poluentes", "🗓️ Sazonalidade",
                "🌡️ Clima x poluição", "🗺️ Mapa", "📋 Tabela dinâmica"])

# ------------------------------- Evolução temporal -------------------------
with abas[0]:
    c1, c2 = st.columns([2, 1])
    nome_ind = c1.selectbox("Indicador", list(INDICADORES), key="ind_tempo")
    janela = c2.slider("Janela da média móvel (meses)", 1, 24, 12)
    ind = INDICADORES[nome_ind]

    serie = d.groupby("data")[ind].mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie.index, y=serie.values, mode="lines", name="Média mensal",
                             line=dict(color=COR_NEUTRA, width=1.5)))
    if janela > 1 and len(serie) >= janela:
        mm = serie.rolling(janela).mean()
        fig.add_trace(go.Scatter(x=mm.index, y=mm.values, mode="lines", name=f"Média móvel de {janela} meses",
                                 line=dict(color=COR_DESTAQUE, width=3)))
    fig.add_hline(y=serie.mean(), line_dash="dash", line_color="black",
                  annotation_text=f"Média da seleção: {fmt(serie.mean(), 2)}", annotation_position="bottom right")
    fig.update_layout(xaxis_title="Mês", yaxis_title=nome_ind, legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(figura_padrao(fig, 430, f"Evolução mensal: {nome_ind} (média das cidades selecionadas)"))

    col_a, col_b = st.columns(2)
    anual_regiao = d.groupby(["ano", "regiao"])[ind].mean().reset_index()
    fig = px.line(anual_regiao, x="ano", y=ind, color="regiao", markers=True, color_discrete_map=CORES_REGIAO,
                  category_orders={"regiao": ORDEM_REGIOES}, labels={"ano": "Ano", ind: nome_ind, "regiao": "Região"})
    fig.update_xaxes(dtick=1)
    col_a.plotly_chart(figura_padrao(fig, 400, f"{nome_ind} por região e ano"))

    niveis_ano = (pd.crosstab(d["ano"], d["nivel_qualidade"], normalize="index") * 100).reset_index() \
        .melt(id_vars="ano", var_name="nivel", value_name="pct")
    fig = px.bar(niveis_ano, x="ano", y="pct", color="nivel", color_discrete_map=CORES_NIVEL,
                 category_orders={"nivel": NIVEIS}, labels={"ano": "Ano", "pct": "% dos registros", "nivel": "Nível"})
    fig.update_xaxes(dtick=1)
    col_b.plotly_chart(figura_padrao(fig, 400, "Níveis de qualidade do ar por ano"))

    anual = d.groupby("ano")["indice_qualidade_ar"].mean()
    if len(anual) >= 3:
        reg = stats.linregress(anual.index, anual.values)
        if reg.pvalue < 0.05:
            veredito = "uma **tendência estatisticamente significativa de " + \
                       ("piora" if reg.slope > 0 else "melhora") + "**"
        else:
            veredito = "**ausência de tendência estatisticamente significativa** (estabilidade)"
        st.info(f"**Interpretação:** o índice médio varia de {fmt(anual.min(), 2)} ({anual.idxmin()}) a "
                f"{fmt(anual.max(), 2)} ({anual.idxmax()}). A reta de tendência indica {fmt(reg.slope, 3)} ponto(s) "
                f"por ano (p = {fmt(reg.pvalue, 2)}), isto é, {veredito}. Oscilações de um ano para o outro "
                f"são maiores que qualquer movimento de longo prazo.")
    else:
        st.info("Selecione ao menos 3 anos para avaliar a tendência do índice.")

# ------------------------------- Cidades e regiões -------------------------
with abas[1]:
    c1, c2, c3 = st.columns([1.2, 1, 1])
    n_cidades = d["cidade"].nunique()
    modo = c1.radio("Ranking", ["Mais poluídas", "Menos poluídas"], horizontal=True)
    n_top = c2.slider("Quantidade de cidades", 1, max(n_cidades, 2), min(10, max(n_cidades, 2)))
    zero = c3.checkbox("Eixo a partir de zero", value=False)

    ranking = media_cidade.sort_values(ascending=(modo == "Menos poluídas")).head(n_top).sort_values(
        ascending=(modo == "Mais poluídas"))
    rank_df = ranking.rename("indice").reset_index()
    rank_df["regiao"] = rank_df["cidade"].map(d.drop_duplicates("cidade").set_index("cidade")["regiao"])
    fig = px.bar(rank_df, x="indice", y="cidade", orientation="h", color="regiao",
                 color_discrete_map=CORES_REGIAO, text=rank_df["indice"].round(2),
                 labels={"indice": "Índice médio de qualidade do ar", "cidade": "Cidade", "regiao": "Região"})
    if not zero and len(rank_df) > 1:
        fig.update_xaxes(range=[rank_df["indice"].min() * 0.97, rank_df["indice"].max() * 1.01])
    fig.add_vline(x=iqa_medio, line_dash="dash", line_color="black")
    st.plotly_chart(figura_padrao(fig, 120 + 32 * len(rank_df),
                                  f"{modo}: índice médio de qualidade do ar (linha tracejada = média da seleção)"))

    col_a, col_b = st.columns(2)
    reg_df = d.groupby("regiao").agg(indice=("indice_qualidade_ar", "mean"), critico=("critico_pct", "mean")).reset_index()
    fig = px.bar(reg_df, x="regiao", y="indice", color="regiao", color_discrete_map=CORES_REGIAO,
                 category_orders={"regiao": ORDEM_REGIOES}, text=reg_df["indice"].round(2),
                 labels={"regiao": "Região", "indice": "Índice médio"})
    fig.update_layout(showlegend=False)
    col_a.plotly_chart(figura_padrao(fig, 380, "Índice médio por região"))
    fig = px.bar(reg_df, x="regiao", y="critico", color="regiao", color_discrete_map=CORES_REGIAO,
                 category_orders={"regiao": ORDEM_REGIOES}, text=reg_df["critico"].round(1),
                 labels={"regiao": "Região", "critico": "% de registros críticos"})
    fig.update_layout(showlegend=False)
    col_b.plotly_chart(figura_padrao(fig, 380, "Registros críticos (Ruim/Péssima) por região"))

    st.markdown("**Distribuição do índice por região** (cada caixa resume todos os registros mensais da região)")
    presentes = [r for r in ORDEM_REGIOES if r in set(d["regiao"])]
    fig_sb, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=d, x="regiao", y="indice_qualidade_ar", hue="regiao", order=presentes, hue_order=presentes,
                palette=CORES_REGIAO, legend=False, ax=ax)
    ax.axhline(iqa_medio, color="black", linestyle="--", linewidth=1)
    ax.set_title("Dispersão do índice de qualidade do ar por região", fontweight="bold")
    ax.set_xlabel("Região")
    ax.set_ylabel("Índice de qualidade do ar")
    sns.despine()
    st.pyplot(fig_sb)
    plt.close(fig_sb)

    amplitude = media_cidade.max() - media_cidade.min()
    dp_mensal = d.groupby("cidade")["indice_qualidade_ar"].std().mean()
    p_cid, p_reg = anova_p(d, "cidade"), anova_p(d, "regiao")
    texto = (f"**Interpretação:** a cidade mais poluída da seleção é **{cidade_top}** "
             f"({fmt(media_cidade.max(), 2)}) e a menos poluída é **{media_cidade.idxmin()}** "
             f"({fmt(media_cidade.min(), 2)}), uma diferença de {fmt(amplitude, 2)} ponto(s). ")
    if not np.isnan(dp_mensal):
        texto += f"Como o índice de uma mesma cidade varia, em média, {fmt(dp_mensal, 1)} ponto(s) de um mês para o outro, "
        texto += "o ranking é um retrato do período e deve ser lido com cautela. "
    if p_cid is not None:
        texto += f"ANOVA entre cidades: p = {fmt(p_cid, 2)} " + \
                 ("(diferença significativa). " if p_cid < 0.05 else "(sem diferença significativa). ")
    if p_reg is not None:
        texto += f"Entre regiões: p = {fmt(p_reg, 2)} " + \
                 ("(diferença significativa)." if p_reg < 0.05 else "(sem diferença significativa).")
    st.info(texto)

# ------------------------------- Poluentes ---------------------------------
with abas[2]:
    freq = (d["poluente_predominante"].value_counts(normalize=True) * 100).reindex(
        [NOMES[p] for p in POLUENTES]).fillna(0)
    part = participacao.reindex([NOMES[p] for p in POLUENTES])
    col_a, col_b = st.columns(2)
    fig = px.bar(x=freq.index, y=freq.values, text=freq.round(1).astype(str) + "%",
                 labels={"x": "Poluente", "y": "% dos registros"})
    fig.update_traces(marker_color=[COR_DESTAQUE if p == freq.idxmax() else COR_NEUTRA for p in freq.index])
    col_a.plotly_chart(figura_padrao(fig, 380, "Frequência como poluente predominante"))
    fig = px.bar(x=part.index, y=part.values, text=part.round(1).astype(str) + "%",
                 labels={"x": "Poluente", "y": "% do índice"})
    fig.update_traces(marker_color=[COR_DESTAQUE if p == part.idxmax() else COR_NEUTRA for p in part.index])
    col_b.plotly_chart(figura_padrao(fig, 380, "Participação média na composição do índice"))

    agrupar = st.radio("Perfil de poluentes por", ["Região", "Estado (UF)", "Cidade"], horizontal=True)
    chave = {"Região": "regiao", "Estado (UF)": "uf", "Cidade": "cidade"}[agrupar]
    relativo = d.groupby(chave)[POLUENTES].mean() / df[POLUENTES].mean() * 100
    relativo.columns = [NOMES[c] for c in relativo.columns]
    if chave == "regiao":
        relativo = relativo.reindex([r for r in ORDEM_REGIOES if r in relativo.index])
    fig = px.imshow(relativo.round(1), text_auto=".0f", color_continuous_scale="RdYlGn_r", color_continuous_midpoint=100,
                    aspect="auto", labels=dict(x="Poluente", y=agrupar, color="Índice relativo"))
    st.plotly_chart(figura_padrao(fig, max(320, 90 + 26 * len(relativo)),
                                  f"Nível de cada poluente por {agrupar.lower()} (100 = média nacional da base completa)"))

    pesos_txt = {NOMES[p]: round(float(df[f"contrib_{p}"].sum() / df[p].sum()), 2) for p in POLUENTES}
    st.info(f"**Interpretação:** o **{poluente_top}** é o poluente predominante, com {fmt(participacao[poluente_top])}% da "
            f"composição do índice e o maior peso em {fmt(freq[poluente_top])}% dos registros. Isso ocorre porque o índice "
            f"é uma soma ponderada das concentrações (pesos: " +
            ", ".join(f"{k} = {fmt(v, 2)}" for k, v in pesos_txt.items()) +
            "). Do ponto de vista ambiental, o material particulado fino é o poluente de maior impacto na saúde, "
            "o que torna o PM2.5 a prioridade natural para o controle de emissões. No mapa de calor, valores acima de 100 "
            "indicam níveis superiores à média nacional daquele poluente.")

# ------------------------------- Sazonalidade ------------------------------
with abas[3]:
    mapa = d.pivot_table(index="ano", columns="mes", values="indice_qualidade_ar", aggfunc="mean")
    mapa.columns = [MESES_PT[m] for m in mapa.columns]
    fig = go.Figure(go.Heatmap(z=mapa.values, x=list(mapa.columns), y=[str(a) for a in mapa.index],
                               colorscale="YlOrRd", colorbar=dict(title="Índice médio"),
                               text=np.round(mapa.values, 1), texttemplate="%{text}",
                               hovertemplate="%{y} - %{x}<br>Índice médio: %{z:.2f}<extra></extra>"))
    fig.update_layout(xaxis_title="Mês", yaxis_title="Ano", yaxis=dict(autorange="reversed"))
    st.plotly_chart(figura_padrao(fig, 130 + 40 * len(mapa), "Heatmap mensal do índice de qualidade do ar"))

    perfil = d.groupby("mes").agg(indice=("indice_qualidade_ar", "mean"), critico=("critico_pct", "mean")).reset_index()
    perfil["mes_nome"] = perfil["mes"].map(MESES_PT)
    col_a, col_b = st.columns(2)
    fig = px.bar(perfil, x="mes_nome", y="indice", text=perfil["indice"].round(1),
                 labels={"mes_nome": "Mês", "indice": "Índice médio"})
    fig.update_traces(marker_color=[COR_DESTAQUE if v == perfil["indice"].max() else COR_NEUTRA for v in perfil["indice"]])
    fig.update_yaxes(range=[perfil["indice"].min() * 0.97, perfil["indice"].max() * 1.01])
    col_a.plotly_chart(figura_padrao(fig, 380, "Índice médio por mês do ano (eixo ajustado)"))
    fig = px.bar(perfil, x="mes_nome", y="critico", text=perfil["critico"].round(1),
                 labels={"mes_nome": "Mês", "critico": "% de registros críticos"})
    fig.update_traces(marker_color=[COR_DESTAQUE if v == perfil["critico"].max() else COR_NEUTRA for v in perfil["critico"]])
    col_b.plotly_chart(figura_padrao(fig, 380, "Registros críticos por mês do ano"))

    serie_n = d.groupby("data")["indice_qualidade_ar"].mean()
    mes_alto, mes_baixo = perfil.loc[perfil["indice"].idxmax(), "mes_nome"], perfil.loc[perfil["indice"].idxmin(), "mes_nome"]
    p_mes = anova_p(d, "mes")
    texto = (f"**Interpretação:** o mês com maior índice médio é **{mes_alto}** ({fmt(perfil['indice'].max(), 2)}) e o menor é "
             f"**{mes_baixo}** ({fmt(perfil['indice'].min(), 2)}); o maior percentual de registros críticos ocorre em "
             f"**{perfil.loc[perfil['critico'].idxmax(), 'mes_nome']}** ({fmt(perfil['critico'].max(), 1)}%). ")
    if p_mes is not None:
        texto += f"A diferença entre os meses " + ("**é**" if p_mes < 0.05 else "**não é**") + \
                 f" estatisticamente significativa (ANOVA, p = {fmt(p_mes, 2)})"
        texto += ". No heatmap, não há faixas de meses persistentemente mais críticas em todos os anos." \
            if p_mes >= 0.05 else ", indicando um padrão sazonal."
    st.info(texto)

    st.markdown("**Decomposição da série temporal** (observado, tendência, sazonalidade e resíduo)")
    if len(serie_n) >= 36 and serie_n.index.is_monotonic_increasing:
        tendencia, sazonal, residuo, f_saz, f_ten = decompor(serie_n)
        fig_dec, eixos = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
        for ax, (nome, vals, cor) in zip(eixos, [("Observado", serie_n, "#34495E"), ("Tendência", tendencia, COR_DESTAQUE),
                                                  ("Sazonalidade", sazonal, COR_APOIO), ("Resíduo", residuo, "#7F8C8D")]):
            ax.plot(vals.index, vals.values, color=cor, linewidth=1.6)
            ax.set_ylabel(nome)
        eixos[0].set_title("Decomposição do índice de qualidade do ar (média da seleção)", fontweight="bold")
        eixos[-1].set_xlabel("Ano")
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig_dec)
        plt.close(fig_dec)
        st.caption(f"Força da sazonalidade: {fmt(f_saz, 2)} · Força da tendência: {fmt(f_ten, 2)} "
                   "(0 = ausente, 1 = dominante). Valores baixos indicam que o resíduo (variação aleatória mensal) "
                   "domina a série.")
    else:
        st.info("A decomposição exige ao menos 36 meses consecutivos de dados. Amplie a seleção de anos e meses.")

# ------------------------------- Clima x poluição --------------------------
with abas[4]:
    c1, c2 = st.columns(2)
    clima_nome = c1.selectbox("Variável climática", ["Temperatura média (°C)", "Umidade relativa (%)"])
    alvo_nome = c2.selectbox("Indicador de poluição", list(INDICADORES), key="ind_clima")
    xcol = "temperatura_media" if clima_nome.startswith("Temp") else "umidade"
    ycol = INDICADORES[alvo_nome]

    if len(d) >= 3 and d[xcol].nunique() > 1:
        r, p = stats.pearsonr(d[xcol], d[ycol])
        rho, _ = stats.spearmanr(d[xcol], d[ycol])
        coef_a, coef_b = np.polyfit(d[xcol], d[ycol], 1)
        xs = np.linspace(d[xcol].min(), d[xcol].max(), 50)
        fig = px.scatter(d, x=xcol, y=ycol, opacity=0.25, hover_data=["cidade", "ano", "mes_nome"],
                         labels={xcol: clima_nome, ycol: alvo_nome})
        fig.update_traces(marker=dict(size=5, color="#5D6D7E"))
        fig.add_trace(go.Scatter(x=xs, y=coef_a * xs + coef_b, mode="lines", name="Tendência linear",
                                 line=dict(color=COR_DESTAQUE, width=3)))
        fig.update_layout(showlegend=False)
        st.plotly_chart(figura_padrao(fig, 460, f"{alvo_nome} x {clima_nome} (cada ponto = uma cidade em um mês)"))

        m1, m2, m3 = st.columns(3)
        m1.metric("Correlação de Pearson (r)", fmt(r, 3))
        m2.metric("Correlação de Spearman (ρ)", fmt(rho, 3))
        m3.metric("p-valor", fmt(p, 3))
        st.info(f"**Interpretação:** a correlação entre {clima_nome.split(' (')[0].lower()} e {alvo_nome} é "
                f"**{forca_correlacao(r)}** (r = {fmt(r, 3)}), " +
                ("estatisticamente significativa" if p < 0.05 else "não significativa") +
                f" (p = {fmt(p, 3)}); a variável climática explica apenas {fmt(100 * r ** 2, 2)}% da variação "
                "do indicador. Quando r é próximo de zero, a reta de tendência é praticamente horizontal e o clima "
                "não ajuda a prever a poluição nesta base.")
    else:
        st.info("Dados insuficientes na seleção atual para calcular a correlação.")

    col_a, col_b = st.columns([1.1, 1])
    colunas_corr = POLUENTES + ["temperatura_media", "umidade", "indice_qualidade_ar"]
    rotulos = {**NOMES, "temperatura_media": "Temperatura", "umidade": "Umidade", "indice_qualidade_ar": "Índice"}
    if len(d) >= 3:
        matriz = d[colunas_corr].corr().rename(index=rotulos, columns=rotulos)
        fig_c, ax = plt.subplots(figsize=(6.5, 5))
        sns.heatmap(matriz, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, linewidths=0.5,
                    cbar_kws={"label": "Pearson"}, ax=ax)
        ax.set_title("Matriz de correlação", fontweight="bold")
        plt.tight_layout()
        col_a.pyplot(fig_c)
        plt.close(fig_c)

        linhas = []
        for clima, rot in [("temperatura_media", "Temperatura"), ("umidade", "Umidade")]:
            for alvo in ["indice_qualidade_ar"] + POLUENTES:
                if d[clima].nunique() > 1 and d[alvo].nunique() > 1:
                    rr, pp = stats.pearsonr(d[clima], d[alvo])
                    linhas.append({"Clima": rot, "Poluição": rotulos[alvo], "r": round(rr, 3),
                                   "p-valor": round(pp, 3), "Significativa (p<0,05)": "sim" if pp < 0.05 else "não"})
        col_b.markdown("**Correlação clima x poluição (com significância)**")
        col_b.dataframe(pd.DataFrame(linhas), hide_index=True)

# ------------------------------- Mapa --------------------------------------
with abas[5]:
    metrica = st.selectbox("Métrica do mapa", ["Índice médio de qualidade do ar", "PM2.5 médio", "% de períodos críticos"])
    col_m = {"Índice médio de qualidade do ar": "indice_qualidade_ar", "PM2.5 médio": "pm25",
             "% de períodos críticos": "critico_pct"}[metrica]
    mapa_df = d.groupby(["cidade", "uf", "regiao"])[col_m].mean().reset_index()
    mapa_df["lat"] = mapa_df["cidade"].map(lambda c: COORDENADAS.get(c, (np.nan, np.nan))[0])
    mapa_df["lon"] = mapa_df["cidade"].map(lambda c: COORDENADAS.get(c, (np.nan, np.nan))[1])
    sem_coord = mapa_df["lat"].isna().sum()
    mapa_df = mapa_df.dropna(subset=["lat", "lon"])
    if mapa_df.empty:
        st.info("Não há coordenadas cadastradas para as cidades da seleção.")
    else:
        mapa_df["valor"] = mapa_df[col_m].round(2)
        fig = px.scatter_map(mapa_df, lat="lat", lon="lon", color="valor", hover_name="cidade",
                             hover_data={"uf": True, "regiao": True, "valor": True, "lat": False, "lon": False},
                             color_continuous_scale="YlOrRd", zoom=3, center={"lat": -14.5, "lon": -52},
                             labels={"valor": metrica}, map_style="open-street-map")
        fig.update_traces(marker=dict(size=16))
        st.plotly_chart(figura_padrao(fig, 600, f"{metrica} por cidade (2015–2024, seleção atual)"))
        if sem_coord:
            st.caption(f"{sem_coord} cidade(s) sem coordenadas cadastradas não aparecem no mapa.")
        top_m = mapa_df.sort_values("valor", ascending=False).iloc[0]
        st.info(f"**Interpretação:** a cidade com o maior valor de \"{metrica.lower()}\" é **{top_m['cidade']} "
                f"({top_m['uf']})**, com {fmt(top_m['valor'], 2)}. As cores variam em faixa estreita, sem "
                "concentração geográfica evidente: não há uma região do país consistentemente mais afetada.")

# ------------------------------- Tabela dinâmica ---------------------------
with abas[6]:
    dimensoes = {"Região": "regiao", "Estado (UF)": "uf", "Cidade": "cidade", "Ano": "ano", "Mês": "mes",
                 "Estação": "estacao", "Nível de qualidade": "nivel_qualidade"}
    valores = {**INDICADORES, "Temperatura média": "temperatura_media", "Umidade": "umidade",
               "% de registros críticos": "critico_pct"}
    funcoes = {"Média": "mean", "Mediana": "median", "Máximo": "max", "Mínimo": "min", "Contagem": "count"}

    c1, c2, c3, c4 = st.columns(4)
    lin_nome = c1.selectbox("Linhas", list(dimensoes), index=0)
    col_nome = c2.selectbox("Colunas", list(dimensoes), index=3)
    val_nome = c3.selectbox("Valores", list(valores), index=0)
    fun_nome = c4.selectbox("Função", list(funcoes), index=0)

    if lin_nome == col_nome:
        st.warning("Escolha dimensões diferentes para linhas e colunas.")
    else:
        pivo = d.pivot_table(index=dimensoes[lin_nome], columns=dimensoes[col_nome], values=valores[val_nome],
                             aggfunc=funcoes[fun_nome])

        def reordenar(eixo, nome_dim):
            if nome_dim == "Estação":
                return [e for e in ORDEM_ESTACOES if e in eixo]
            if nome_dim == "Nível de qualidade":
                return [n for n in NIVEIS if n in eixo]
            if nome_dim == "Região":
                return [r for r in ORDEM_REGIOES if r in eixo] + [r for r in eixo if r not in ORDEM_REGIOES]
            return list(eixo)

        pivo = pivo.reindex(index=reordenar(pivo.index, lin_nome), columns=reordenar(pivo.columns, col_nome))
        if lin_nome == "Mês":
            pivo.index = [MESES_PT[m] for m in pivo.index]
        if col_nome == "Mês":
            pivo.columns = [MESES_PT[m] for m in pivo.columns]
        pivo.columns = [str(c) for c in pivo.columns]
        pivo.index.name, pivo.columns.name = lin_nome, col_nome

        casas = "{:.0f}" if fun_nome == "Contagem" else "{:.2f}"
        st.dataframe(pivo.style.background_gradient(cmap="YlOrRd", axis=None).format(casas, na_rep="-"))
        st.download_button("Baixar tabela dinâmica (CSV)", pivo.to_csv(sep=";", decimal=",").encode("utf-8-sig"),
                           file_name="tabela_dinamica.csv", mime="text/csv")
        st.info(f"**Interpretação:** a tabela mostra a {fun_nome.lower()} de \"{val_nome.lower()}\" por "
                f"{lin_nome.lower()} e {col_nome.lower()}. O degradê destaca as células de maior valor; compare a "
                "variação dentro de uma mesma linha ou coluna com a diferença entre grupos para avaliar se o padrão "
                "é consistente ou apenas oscilação mensal.")

    with st.expander("Ver e baixar os dados filtrados"):
        mostrar = ["data", "regiao", "uf", "cidade", "pm25", "pm10", "no2", "co", "o3", "temperatura_media",
                   "umidade", "indice_qualidade_ar", "nivel_qualidade", "poluente_predominante"]
        st.dataframe(d[mostrar].sort_values(["data", "cidade"]), hide_index=True)
        st.download_button("Baixar dados filtrados (CSV)", d[mostrar].to_csv(index=False, sep=";", decimal=",")
                           .encode("utf-8-sig"), file_name="dados_filtrados.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# Conclusão executiva
# ---------------------------------------------------------------------------
st.header("Conclusão executiva")
col_a, col_b = st.columns(2)
with col_a.container(border=True):
    st.subheader("Resultados da base completa")
    st.markdown(
        """
* **Índice médio:** 67,74, com **11,8%** dos registros cidade-mês em nível Ruim. Nenhum registro atingiu o nível Péssima.
* **Poluente predominante: PM2.5**, com cerca de 47% da composição do índice e o maior peso em mais de 93% dos registros.
* **Cidades que mais exigem atenção:** Florianópolis e Joinville, que combinam os maiores índices médios e os maiores percentuais de registros críticos; Petrópolis e São Paulo vêm em seguida.
* **Sem evidência estatística** de sazonalidade, tendência de piora ou melhora, diferença entre regiões ou relação com temperatura e umidade. As variações observadas são compatíveis com oscilação aleatória mensal.
* **Ressalva:** a base é simulada e mensal, sem unidades de medida; os resultados descrevem os dados fornecidos e não a situação real das cidades.
        """
    )
with col_b.container(border=True):
    st.subheader("Leitura da seleção atual")
    pior_mes = d.groupby("mes")["critico_pct"].mean().idxmax()
    st.markdown(
        f"""
* **Cidade mais poluída:** {cidade_top} ({fmt(media_cidade.max(), 2)}); **região mais afetada:** {regiao_top}.
* **Poluente predominante:** {poluente_top} ({fmt(participacao[poluente_top])}% do índice).
* **Períodos críticos:** {fmt(pct_critico)}% dos registros ({fmt(pct_critico - crit_base)} p.p. em relação à base completa); mês com maior percentual: {MESES_PT[pior_mes]}.
* **Decisão sugerida:** priorizar o monitoramento e o controle de emissões de {poluente_top} nas cidades de maior índice, mantendo vigilância contínua ao longo do ano enquanto não houver dados que comprovem um padrão sazonal.
        """
    )

st.divider()
st.caption(f"**{TITULO}** · {DISCIPLINA} · Professor: {PROFESSOR} · Aluno: {ALUNO} · "
           "Base de dados simulada, fornecida para fins acadêmicos.")
