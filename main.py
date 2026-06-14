from src.carregar_dados import carregar_dados
from src.preprocessamento import Preprocessador 
from src.treinamento import Treinador
from src.descricao import descrever_cluster

if __name__ == '__main__':
    # carregamento
    dados = carregar_dados()
    # print(f'[DEBUG] Dados Preview:\n{dados.head(10)}\n')
    # print(f'[DEBUG] Tipos de dados:\n{dados.dtypes}\n')
    
    # instancia da classe de preprocessamento
    preprocessador = Preprocessador(dados)
    
    # tratamento de dados: remoção de colunas
    dados_num = preprocessador.tratar_dados()
    
    # normalizacao dos dados
    dados_num_norm = preprocessador.normalizar(dados_num)
    # print(f'[DEBUG] Dados numéricos normalizados Preview:\n{dados_num_norm}\n')
    
    # treinamento de dados
    treinador = Treinador(dados_num_norm)
    modelo_treinado = treinador.treinar_modelo()
    print(f'Modelo treinado:\n{modelo_treinado}')
    
    # descrição de dados
    descrever_cluster()