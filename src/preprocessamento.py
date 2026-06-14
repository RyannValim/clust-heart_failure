import pandas as pd

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from src.salvar_modelo import salvar_modelo

class Preprocessador():
    def __init__(self, dados):
        self.dados = dados
    
    def tratar_dados(self):
        # copia para não perder dados por referência
        copia_dados = self.dados.copy()
        
        # retorna dados removendo coluna time: dado não-clínico e DEATH_EVENT: target
        return copia_dados.drop(columns=['time', 'DEATH_EVENT'])
    
    def normalizar(self, dados):
        # colunas numéricas: receberão MinMaxScaler
        cols_numericas = [
            'age',
            'creatinine_phosphokinase',
            'ejection_fraction',
            'platelets',
            'serum_creatinine',
            'serum_sodium'
        ]
        
        # colunas binárias booleanas: já estão em 0/1, passam sem transformação
        cols_binarias = [
            'anaemia',
            'diabetes',
            'high_blood_pressure',
            'smoking'
        ]

        # instancia e fita o scaler apenas nas colunas numéricas
        scaler = MinMaxScaler()
        scaler_norm = scaler.fit(dados[cols_numericas])
        salvar_modelo(scaler_norm, 'Normalizador_MinMaxScaler')

        # normaliza apenas as colunas numéricas
        df_numericas_norm = pd.DataFrame(
            data=scaler_norm.transform(dados[cols_numericas]),
            columns=cols_numericas
        )

        # reseta o index das binárias para concatenar corretamente
        df_binarias = dados[cols_binarias].reset_index(drop=True)

        # mapeia sex para texto antes do OHE: 1=Masculino, 0=Feminino (confirmado: 194 masc, 105 fem)
        df_sex = dados['sex'].map({1: 'Masculino', 0: 'Feminino'}).reset_index(drop=True).to_frame()

        # aplica OHE em sex e salva o encoder
        ohe = OneHotEncoder(sparse_output=False)
        ohe_norm = ohe.fit(df_sex)
        salvar_modelo(ohe_norm, 'OHE_Sex')

        # gera as colunas sex_Feminino e sex_Masculino
        df_sex_ohe = pd.DataFrame(
            data=ohe_norm.transform(df_sex),
            columns=ohe_norm.get_feature_names_out(['sex'])
        )

        # concatena numéricas normalizadas + binárias intactas + sex OHE
        df_final = pd.concat([df_numericas_norm, df_binarias, df_sex_ohe], axis=1)
        
        # salva o dataset normalizado para uso futuro
        df_final.to_csv('./datasets/heart_failure_clinical_records_dataset_norm.csv', index=False)

        # retorna df com numéricas normalizadas + binárias intactas + sex expandido
        return df_final