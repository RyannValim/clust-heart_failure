
# Clusterizador do dataset Heart Failure

Sistema de clusterização do dataset UCI Heart Failure Clinical Records usando KMeans, para atribuir pacientes desconhecidos a um perfil clínico (cluster) com base em 11 features clínicas.

## Stack

- Python 3.14
- scikit-learn (KMeans, MinMaxScaler, OneHotEncoder)
- pandas, numpy, matplotlib, scipy

## Como rodar

### 1. Criar e ativar o ambiente virtual

```powershell
python -m venv .venv
.venv\Scripts\activate
```

No Linux: `source .venv/bin/activate`

### 2. Instalar dependências

```powershell
pip install -r requirements.txt
```

### 3. Executar o treinamento e a descrição dos clusters

```powershell
python main.py
```

Isso vai: carregar o dataset, pré-processar, treinar o KMeans com K=24 (determinado pelo elbow curve), salvar os modelos em `models/` e imprimir o perfil dos 24 clusters.

### 4. Executar a inferência de um paciente novo

```powershell
python inferencia.py
```

O paciente sintético dentro de `inferencia.py` será atribuído a um cluster, e o perfil daquele cluster será impresso.

### 5. (Opcional) Comparar métricas para diferentes valores de K

```powershell
python comparar_metricas.py
```

Roda Silhouette, Calinski-Harabasz e Davies-Bouldin para K de 2 a 100, imprime a tabela e salva o gráfico em `plots/`.

## Estrutura

```
clust-heart_failure/
├── datasets/       # CSV original e normalizado
├── models/         # .pkl gerados pelo treinamento
├── plots/          # Elbow curve e gráficos das métricas
├── src/            # Módulos do pipeline
│   ├── carregar_dados.py
│   ├── preprocessamento.py
│   ├── treinamento.py
│   ├── descricao.py
│   ├── predicao.py
│   └── salvar_modelo.py
├── main.py             # Fluxo completo de treinamento
├── inferencia.py       # Inferência standalone (precisa do main.py rodado antes)
└── comparar_metricas.py
```

## Decições técnicas

Documentadas em `notas.txt`.
