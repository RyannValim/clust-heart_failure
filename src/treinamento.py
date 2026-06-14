import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from src.salvar_modelo import salvar_modelo

class Treinador():
    def __init__(self, dados_num_norm):
        self.dados_num_norm = dados_num_norm
    
    def treinar_modelo(self):
        # obtem k_otimo com o cálculo do elbow curve
        k_otimo = self.calc_elbow()
        
        # treina com K ótimo
        treinador_heart_failure = KMeans(n_clusters=k_otimo, random_state=42).fit(self.dados_num_norm)
        
        # salva o modelo treinador
        salvar_modelo(modelo=treinador_heart_failure, nome_modelo="Treinador_KMeans")
        
        # retorna o modelo treinado
        return treinador_heart_failure
    
    def calc_elbow(self):
        distorcoes = []
        K = range(1, 101)
        for i in K:
            treinador = KMeans(n_clusters=i, random_state=42).fit(self.dados_num_norm)
            distorcoes.append(
                sum(np.min(cdist(self.dados_num_norm, treinador.cluster_centers_, 'euclidean'),
                           axis=1) / self.dados_num_norm.shape[0])
            )
        
        # calcula a distância de cada ponto da curva ao segmento que liga o primeiro ao último ponto
        # o K com maior distância ao segmento é o joelho geométrico da curva
        distancias = []
        for i in range(len(distorcoes)):
            x = K[i]
            y = distorcoes[i]
        
            x0 = K[0]
            y0 = distorcoes[0]
            xn = K[-1]
            yn = distorcoes[-1]
            
            distancias.append(abs(((
                (yn - y0)*x) - (xn - x0)*y) + (xn * y0) - (yn * x0)
            ) / sqrt((yn - y0)**2 + (xn - x0)**2))
        
        # plotagem do elbow curve
        plt.plot(K, distorcoes)
        plt.savefig('./plots/Heart_Failure_Elbow_Curve.png')
        plt.close()
        
        # retorna K ótimo
        return K[distancias.index(np.max(distancias))]