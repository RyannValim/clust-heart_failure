from src.predicao import prever
from src.descricao import descrever_cluster

if __name__ == '__main__':
    # dados sintéticos para teste
    # cluster 0: homem, não anêmico, não diabético, sem hipertensão, fumante, CPK alto
    cluster_0  = [59, 1050, 37, 245000, 1.3, 136, 0, 0, 0, 1, 1]

    # cluster 4: homem, anêmico, sem diabetes, hipertenso, fumante, idoso, creatinina alta
    cluster_4  = [70, 473, 44, 287000, 2.5, 137, 1, 0, 1, 1, 1]

    # cluster 9: homem, anêmico, diabético, sem hipertensão, não fumante, creatinina elevada
    cluster_9  = [59, 400, 37, 249000, 1.7, 136, 1, 1, 0, 0, 1]

    # cluster 13: homem, anêmico, sem diabetes, sem hipertensão, fumante
    cluster_13 = [58, 251, 36, 279000, 1.1, 138, 1, 0, 0, 1, 1]

    # cluster 21: mulher, anêmica, diabética, sem hipertensão, fumante, creatinina muito alta
    cluster_21 = [60, 260, 38, 255000, 2.2, 132, 1, 1, 0, 1, 0]

    # cluster 22: homem, não anêmico, diabético, hipertenso, não fumante, CPK muito alto
    cluster_22 = [61, 1132, 45, 271000, 1.0, 138, 0, 1, 1, 0, 1]

    # cluster 23: homem, anêmico, diabético, sem hipertensão, fumante
    cluster_23 = [61, 277, 36, 238000, 1.1, 136, 1, 1, 0, 1, 1]
    
    # inserção manual minha para testes
    paciente_novo = [28, 299, 42, 231000, 1.9, 128, 1, 0, 1, 0, 0]

    # inferência: atribui o paciente a um cluster
    cluster = prever(paciente_novo) # (<- mudar aqui para testar os outros pacientes)
    print(f'Este paciente pertence ao cluster: {cluster[0]}')

    # descrição do cluster atribuído
    print(f'\nPerfil do cluster {cluster[0]}:')
    descrever_cluster(cluster[0])