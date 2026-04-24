# Previsão de Geração de Energia Fotovoltaica com Redes Neurais Artificiais

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

<p align="center">
  Análise comparativa de arquiteturas MLP para previsão de séries temporais de energia solar em <strong>três horizontes de tempo</strong>, avaliando o impacto de diferentes otimizadores, estratégias de batch e configurações de camadas ocultas.
</p>

---

## Índice

- [Contexto do Problema](#-contexto-do-problema)
- [Dataset](#-dataset)
- [Metodologia](#-metodologia)
- [Arquitetura da Rede Neural](#-arquitetura-da-rede-neural)
- [Experimentos](#-experimentos)
- [Resultados](#-resultados)
- [Como Reproduzir](#-como-reproduzir)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Tecnologias](#-tecnologias)

---

## Contexto do Problema

A previsão precisa da geração de energia fotovoltaica é fundamental para o planejamento e a operação eficiente de redes elétricas. Variações climáticas introduzem alta volatilidade na produção, tornando a tarefa desafiadora para modelos tradicionais.

Este projeto utiliza um **Perceptron Multicamadas (MLP)** para modelar a série temporal de produção de uma usina fotovoltaica localizada em **Tauá – Ceará**, avaliando comparativamente diferentes configurações de treinamento em três horizontes de previsão.

---

## Dataset

| Atributo | Descrição |
|---|---|
| **Fonte** | Usina fotovoltaica em Tauá – CE |
| **Variável alvo** | Produção energética diária (kWh) |
| **Tamanho** | ~1000 amostras diárias |
| **Divisão** | 70% treino / 30% teste |
| **Pré-processamento** | Média móvel (janela=3) + normalização Min-Max |

A média móvel é aplicada para suavizar a série temporal antes do treinamento, reduzindo o impacto de outliers pontuais na aprendizagem do modelo.

```
Série bruta  ──► Média Móvel (janela=3) ──► Normalização [0,1] ──► Janelas deslizantes
```

---

## Metodologia

### Horizontes de Previsão

Foram definidos três cenários de acordo com a quantidade de amostras passadas (entradas) e futuras (saídas) utilizadas:

| Caso | Horizonte | Entradas (`n`) | Saídas (`i`) |
|---|---|---|---|
| Caso 01 | Curto Prazo | 3 amostras | 1 amostra |
| Caso 02 | Médio Prazo | 30 amostras | 7 amostras |
| Caso 03 | Longo Prazo | 90 amostras | 30 amostras |

### Variáveis Avaliadas

Para cada horizonte, foram realizados experimentos variando sistematicamente:

- **Neurônios por camada oculta:** 5, 20 e 100
- **Estratégia de batch:** Batch completo, Estocástico (SGD) e Mini-batch (32)
- **Otimizadores:** SGD, SGD+Momentum, Adam, RMSProp, Adagrad, Adadelta, Adamax, Nadam, FTRL

---

## Arquitetura da Rede Neural

```
Entrada          1ª Camada Oculta    2ª Camada Oculta    Saída
─────────        ────────────────    ────────────────    ──────
p(k)    ──┐
p(k-1)  ──┤──►  [N neurônios]  ──►  [N neurônios]  ──►  p(k+1)
p(k-2)  ──┤      ReLU                ReLU                p(k+2)
  ⋮     ──┤                                               ⋮
p(k-n)  ──┘                                              p(k+i)

N ∈ {5, 20, 100}   |   Saída: ativação linear   |   Loss: MSE
```

Todas as redes foram treinadas com:
- **Early Stopping** monitorando `val_loss` (evita overfitting)
- **Validação:** 10% do conjunto de treino
- **Máximo de épocas:** 200 (varredura) / 1000 (modelo final)

---

## Experimentos

O processo de varredura total resultou em **81 configurações por horizonte** (3 tamanhos de rede × 3 estratégias de batch × 9 otimizadores), totalizando **243 modelos treinados**.

```python
# Estrutura da varredura
for n_neuronios in [5, 20, 100]:
    for estrategia_batch in ['Batch', 'Stochastic', 'Mini-batch']:
        for otimizador in lista_otimizadores:
            # treinar → avaliar → registrar MAPE, MAE, MSE
```

---

## Resultados

### Melhores configurações por horizonte

| Horizonte | Otimizador | Neurônios | Batch | MAPE (%) | MSE |
|---|---|---|---|---|---|
| Curto Prazo | Nadam | 20 | Estocástico | 5,5057 | 0,0071 |
| Médio Prazo | Adam | 100 | Estocástico | 8,7125 | 0,0168 |
| Longo Prazo | SGD + Momentum | 20 | Estocástico | 9,4976 | 0,0196 |

### Principais achados

- **Horizonte de curto prazo** obteve os menores erros, como esperado — menor incerteza acumulada ao longo do tempo
- **Otimizadores adaptativos** (Adam, Nadam) tenderam a superar o SGD simples em convergência
- **Estratégia estocástica** (batch=1) apresentou melhor generalização na maioria dos casos, porém com maior tempo de treinamento
- **Número de neurônios** teve impacto diferente por horizonte: redes maiores ajudaram no longo prazo, mas adicionaram ruído no curto

---

## Como Reproduzir

### 1. Clonar o repositório

```bash
git clone https://github.com/victorgmsds/IA_atv2.git
cd IA_atv2
```

### 2. Criar ambiente virtual
```bash
python -m venv venv
source venv/bin/activate #Linux/Mac
venv\Scripts\activate #Windows
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Executar o notebook

```bash
jupyter notebook atv2.ipynb
```

Execute todas as células em ordem (`Kernel → Restart & Run All`).

---

## Estrutura do Repositório

```
📦 nome-do-repositorio
 ┣ 📓 atv2.ipynb          # Notebook principal com todo o pipeline
 ┣ 📄 Dados.mat            # Série temporal da usina
 ┣ 📄 requirements.txt     # Dependências do projeto
 ┗ 📄 README.md            # Este arquivo
```

---

## Tecnologias

| Biblioteca | Uso |
|---|---|
| `TensorFlow / Keras` | Construção e treinamento das redes neurais |
| `Scikit-learn` | Divisão treino/teste e normalização |
| `NumPy` | Manipulação de arrays e janelas deslizantes |
| `Pandas` | Média móvel e organização dos resultados |
| `Matplotlib / Seaborn` | Visualizações e análises comparativas |
| `SciPy` | Carregamento do arquivo `.mat` |

---

## Contexto Acadêmico

Projeto desenvolvido para a disciplina de **Inteligência Artificial**, com foco em redes neurais artificiais aplicadas à previsão de séries temporais de energia renovável.

