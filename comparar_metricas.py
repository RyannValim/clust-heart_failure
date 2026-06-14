import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from src.carregar_dados import carregar_dados
from src.preprocessamento import Preprocessador

def comparar_metricas(dados_num_norm, k_min=2, k_max=101):
    resultados = []
    
    for k in range(k_min, k_max):
        # treina um KMeans para cada K candidato
        modelo = KMeans(n_clusters=k, random_state=42).fit(dados_num_norm)
        labels = modelo.labels_
        
        # calcula as três métricas
        silhouette = silhouette_score(dados_num_norm, labels)
        calinski   = calinski_harabasz_score(dados_num_norm, labels)
        davies     = davies_bouldin_score(dados_num_norm, labels)
        
        resultados.append({
            'K':                 k,
            'Silhouette':        round(silhouette, 4),
            'Calinski-Harabasz': round(calinski, 4),
            'Davies-Bouldin':    round(davies, 4)
        })
    
    # monta e imprime a tabela de comparação
    df_resultados = pd.DataFrame(resultados).set_index('K')
    print(f'Comparação de métricas por K:\n\n{df_resultados.to_string()}')

    # k ótimo por métrica
    k_silhouette = df_resultados['Silhouette'].idxmax()
    k_calinski   = df_resultados['Calinski-Harabasz'].idxmax()
    k_davies     = df_resultados['Davies-Bouldin'].idxmin()

    print(f'\nSilhouette:        k_otimo={k_silhouette}')
    print(f'Calinski-Harabasz: k_otimo={k_calinski}')
    print(f'Davies-Bouldin:    k_otimo={k_davies}')
    
    return df_resultados

def plotar_metricas(df_resultados):
    # plotagem das três métricas em subplots separados
    _, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    axes[0].plot(df_resultados.index, df_resultados['Silhouette'], marker='o')
    axes[0].set_title('Silhouette Score (maior = melhor)')
    axes[0].set_xlabel('K')
    axes[0].set_ylabel('Silhouette')
    
    axes[1].plot(df_resultados.index, df_resultados['Calinski-Harabasz'], marker='o', color='orange')
    axes[1].set_title('Calinski-Harabasz (maior = melhor)')
    axes[1].set_xlabel('K')
    axes[1].set_ylabel('Calinski-Harabasz')
    
    axes[2].plot(df_resultados.index, df_resultados['Davies-Bouldin'], marker='o', color='red')
    axes[2].set_title('Davies-Bouldin (menor = melhor)')
    axes[2].set_xlabel('K')
    axes[2].set_ylabel('Davies-Bouldin')
    
    plt.tight_layout()
    plt.savefig('./plots/Heart_Failure_Metricas.png')
    plt.close()

if __name__ == '__main__':
    # carrega o dataset original e aplica o mesmo preprocessamento do treino
    dados = carregar_dados()
    preprocessador = Preprocessador(dados)
    dados_num = preprocessador.tratar_dados()
    dados_num_norm = preprocessador.normalizar(dados_num)
    
    df_resultados = comparar_metricas(dados_num_norm)
    plotar_metricas(df_resultados)