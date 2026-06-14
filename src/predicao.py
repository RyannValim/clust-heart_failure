import pandas as pd
from pickle import load

def prever(dados_novos):
    # recuperando e instanciando os modelos
    scaler  = load(open('models/HeartFailure_Normalizador_MinMaxScaler.pkl', 'rb'))
    treinador = load(open('models/HeartFailure_Treinador_KMeans.pkl', 'rb'))
    sex_ohe = load(open('models/HeartFailure_OHE_Sex.pkl', 'rb'))

    # recuperando as colunas e montando na ordem da lista recebida
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
    
    todas_cols = cols_numericas + cols_binarias + ['sex']
    
    # montagem do dataframe
    df_dados_novos = pd.DataFrame(
        data=[dados_novos],
        columns=todas_cols
    )
    
    # normaliza as colunas numéricas com o scaler do treino
    df_num_norm = pd.DataFrame(
        data=scaler.transform(df_dados_novos[cols_numericas]),
        columns=cols_numericas
    )
    
    # binárias passam sem transformação
    df_bin = df_dados_novos[cols_binarias].reset_index(drop=True)
    
    # mapeia sex e aplica OHE no treino
    df_sex = df_dados_novos['sex'].map({1: 'Masculino', 0: 'Feminino'}).to_frame()
    df_sex_ohe = pd.DataFrame(
        data=sex_ohe.transform(df_sex),
        columns=sex_ohe.get_feature_names_out(['sex'])
    )
    
    # concatena na mesma ordem do treino
    df_final = pd.concat([df_num_norm, df_bin, df_sex_ohe], axis=1, ignore_index=False)
    
    # retorna o cluster predito
    return treinador.predict(df_final)