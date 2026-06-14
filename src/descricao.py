import pandas as pd
from pickle import load

def descrever_cluster(i=None):
    # load nos modelos salvos com pickle
    treinador_KMeans = load(open('models/HeartFailure_Treinador_KMeans.pkl', 'rb'))
    minMaxScaler = load(open('models/HeartFailure_Normalizador_MinMaxScaler.pkl', 'rb'))
    ohe_sex = load(open('models/HeartFailure_OHE_Sex.pkl', 'rb'))
    
    # colunas na ordem correta do treino
    cols_numericas = [
        'age',
        'creatinine_phosphokinase',
        'ejection_fraction',
        'platelets',
        'serum_creatinine',
        'serum_sodium'
    ]

    cols_binarias = [
        'anaemia',
        'diabetes',
        'high_blood_pressure',
        'smoking'
    ]

    # colunas geradas pelo OHE de sex
    cols_sex_ohe = ohe_sex.get_feature_names_out(['sex']).tolist()

    # define os clusters a descrever: um específico ou todos
    clusters_alvo = [i] if i is not None else range(treinador_KMeans.n_clusters)
    
    # loop de descrição dos clusters
    for c in clusters_alvo:
        # carrega o centroide da vez
        centroide = treinador_KMeans.cluster_centers_[c]
        
        # monta o dataframe com o centroide usando todas as colunas na ordem do treino
        df_centroide = pd.DataFrame(
            data=centroide.reshape(1, -1),
            columns=cols_numericas + cols_binarias + cols_sex_ohe
        )
        
        # desnormalização das colunas numéricas com inverse_transform
        df_numericas_desnorm = pd.DataFrame(
            data=minMaxScaler.inverse_transform(df_centroide[cols_numericas]),
            columns=cols_numericas
        )

        # cabeçalho só quando descreve todos
        if i is None:
            print(f'===== CLUSTER {c} =====')

        # print das numéricas desnormalizadas
        for col in cols_numericas:
            print(f'  {col}: {df_numericas_desnorm[col].values[0]:.2f}')
        
        # print das binárias arredondadas para 0 ou 1
        for col in cols_binarias:
            print(f'  {col}: {"Sim" if round(df_centroide[col].values[0]) == 1 else "Não"}')

        # identifica o sexo do centróide via coluna OHE de maior valor
        idx_sex = df_centroide[cols_sex_ohe].values[0].argmax()
        sex = cols_sex_ohe[idx_sex].split('_', 1)[1]
        print(f'  sex: {sex}')