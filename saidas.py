"""
    Input (pesosIniciais, pesosFinais, etc.) é um array de listas (cada lista representa uma camada -> dentro dessa lista há n listas representando os neurônios e suas ligações com a próxima camada)
    ex: 63 neuronios = 63 listas com 15 pesos (15 ligações com a camada escondida)
    ex2: 15 neurônios = 15 listas com 7 pesos (7 ligações com a camada de saída)
"""
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs(os.path.dirname("saidas/"), exist_ok=True)                                                              # Cria diretório para saídas se não existir

"""
    Salvar parâmetros de arquitetura da mlp em arquivo txt
"""
def salvar_parametros(n_input, n_hidden, n_output, taxa_aprendizado, condicao_parada):
    txt_parametros = open("saidas/parametros.txt", "w")

    txt_parametros.write("Número de neurônios da camada de entrada: " + str(n_input) + "\n")
    txt_parametros.write("Número de neurônios da camada escondida: " + str(n_hidden) + "\n")
    txt_parametros.write("Número de neurônios da camada de saída: " + str(n_output) + "\n")
    txt_parametros.write("Taxa de aprendizado do algoritmo: " + str(taxa_aprendizado) + "\n")
    txt_parametros.write("Condição de parada do algoritmo (em média de erro): " + str(condicao_parada) + "\n")

    txt_parametros.close()

"""
    Salvar pesos iniciais dos neurônios em arquivo txt
"""
def salvar_pesosIniciais(pesosIniciais):
    txt_pesosIniciais = open("saidas/pesos_iniciais.txt", "w")

    j = 0
    for camada in pesosIniciais:
        neuronios = camada.tolist()

        if j == 0:
            txt_pesosIniciais.write("------------- CAMADA SENSORIAL -> ESCONDIDA ------------- \n\n")
        if j == 1:
            txt_pesosIniciais.write("------------- CAMADA ESCONDIDA -> SAIDA -------------- \n\n")

        i = 1
        for pesos in neuronios:
            txt_pesosIniciais.write("Neurônio " + str(i) + ": \n")

            for peso in pesos:
                txt_pesosIniciais.write(str(peso))
                txt_pesosIniciais.write(" ")

            i += 1
            txt_pesosIniciais.write("\n\n")
        j += 1

    txt_pesosIniciais.close()

"""
    Salvar pesos finais dos neurônios em arquivo txt
"""
def salvar_pesosFinais(pesosFinais):
    txt_pesosFinais = open("saidas/pesos_finais.txt", "w")

    j = 0
    for camada in pesosFinais:
        neuronios = camada.tolist()

        if j == 0:
            txt_pesosFinais.write("------------- CAMADA SENSORIAL -> ESCONDIDA ------------- \n\n")
        if j == 1:
            txt_pesosFinais.write("------------- CAMADA ESCONDIDA -> SAIDA -------------- \n\n")

        i = 1
        for pesos in neuronios:
            txt_pesosFinais.write("Neurônio " + str(i) + ": \n")

            for peso in pesos:
                txt_pesosFinais.write(str(peso))
                txt_pesosFinais.write(" ")

            i += 1
            txt_pesosFinais.write("\n\n")
        j += 1

    txt_pesosFinais.close()

"""
    Salva média de erros por época em arquivo txt
"""
def salvar_erroPorEpoca(erro, epoca):
    txt_erroPorEpoca = open("saidas/erro_por_epoca.txt", "a")
    txt_erroPorEpoca.write("Média de erro da época " + str(epoca) + ": " + str(erro) + "\n")
    txt_erroPorEpoca.close()

"""
    Plota gráfico de erros por época
    Entrada: Lista de erros
"""
def plotar_erroPorEpoca(x):
    plt.rcParams['figure.figsize'] = (11, 7)
    plt.title("Erro médio por Época")
    plt.plot(x)
    plt.xlabel('Época')
    plt.ylabel('Erro médio')
    plt.savefig('saidas/ErroPorEpoca.png')

"""
    Salva resultado de cada arquivo teste usado e suas matrizes de confusão
    Entrada: Listas de ouput final de testes, as labels e nome do arquivo para ser salvo separadamente
"""
def salvar_resultadoPorArquivo(testes, labels, arquivo):
    if arquivo is not None:
        txt_resultado_por_arquivo = open("saidas/resultado_por_arquivo_" + arquivo + ".txt", "w")
    else:
        txt_resultado_por_arquivo = open("saidas/resultado_por_arquivo.txt", "w")

    letras = {                                                                                              # Dicionário criado para achar letras por resultado concatenado dos neurônios
        "A": "1000000",
        "B": "0100000",
        "C": "0010000",
        "D": "0001000",
        "E": "0000100",
        "J": "0000010",
        "K": "0000001"
    }

    i = 0
    caracteres_esperados = []
    caracteres_obtidos = []
    letra_codigo = ""

    for teste in range(21):
        # --------------- Gravar resultado esperado (label) dado um teste ------------
        txt_resultado_por_arquivo.write("Resultado esperado " + str(i) + ": ")

        for neuronio in range(7):
            txt_resultado_por_arquivo.write("{0} ".format(labels[teste][neuronio]))
            letra_codigo += str(labels[teste][neuronio])                                                    # Concatena cada resposta esperada de neuronio da lista de labels

        for key, value in letras.items():
            if letra_codigo == value:
                txt_resultado_por_arquivo.write("    --> " + key)                                           # Usa dicionário para achar a letra correspondente ao código de resposta concatenado anteriormente
                caracteres_esperados.append(key)

        txt_resultado_por_arquivo.write("\n")

        letra_codigo = ""

        # --------------- Gravar resultado obtido pela rede neural dado um teste ----------------
        txt_resultado_por_arquivo.write("Resultado obtido " + str(i) + ": ")
        for neuronio in range(7):
            txt_resultado_por_arquivo.write("{0} ".format(round(testes[teste][neuronio])))                  # Concatena resultado dos neurônios para checar no dicionário de letras se existe a letra, dado o resultado.
            letra_codigo += str(round(testes[teste][neuronio]))

        if letra_codigo not in letras.values():                                                             # Marca neurônios não reconhecidos por nenhuma letra no dicionário, tanto para não quebrar a matriz de confusão, quanto para apresentar no resultado por arquivo
            txt_resultado_por_arquivo.write("      --> Não reconhece")
            caracteres_obtidos.append("None")
        else:
            for key, value in letras.items():
                if letra_codigo == value:
                    txt_resultado_por_arquivo.write("      --> " + key)                                     # Procura a letra no dicionário pelo código concatenado anteriormente
                    caracteres_obtidos.append(key)

        txt_resultado_por_arquivo.write("\n")

        # ---------------- Gravar resultado detalhado da rede neural dado um teste ------------------
        txt_resultado_por_arquivo.write("Resultado detalhado " + str(i) + ": ")                             # Apresenta resultado detalhado (isto é, com mais casas decimais) dos neurônios para analisar melhor os dados
        for neuronio in range(7):
            txt_resultado_por_arquivo.write("{:.3f} ".format(testes[teste][neuronio]))
        txt_resultado_por_arquivo.write("\n\n")

        letra_codigo = ""
        i += 1

    # ------------ Criar matriz de confusão e arquivo txt da matrix ---------------
    matrizConfusao = criarMatrizConfusao(caracteres_obtidos, caracteres_esperados)
    print(matrizConfusao)

    if arquivo is not None:
        txt_matrizConfusao_por_arquivo = open("saidas/matrizConfusao_por_arquivo_" + arquivo + ".txt", "w")
    else:
        txt_matrizConfusao_por_arquivo = open("saidas/matrizConfusao_por_arquivo.txt", "w")

    txt_matrizConfusao_por_arquivo.write("  A  B  C  D  E  J  K           --> [Esperado]\n")
    txt_matrizConfusao_por_arquivo.write(str(matrizConfusao))

    txt_resultado_por_arquivo.close()
    txt_matrizConfusao_por_arquivo.close()

"""
    Cria matriz de confusão --> Linha superior = labels, Linha esquerda = obtidos
    Entrada: Listas de caracteres obtidos/esperados(labels) dos 21 testes
"""
def criarMatrizConfusao(caracteres_obtidos, caracteres_esperados):
    labels = ["A", "B", "C", "D", "E", "J", "K"]
    matrizConfusao = np.zeros((7, 7))
    adicionar_i = 0
    adicionar_j = 0

    #   A B C D E J K                 [ESPERADO = EM CIMA, OBTIDO = LADO ESQUERDO]
    # A X X X X X X X
    # B X X X X X X X
    # C X X X X X X X
    # D X X X X X X X
    # E X X X X X X X
    # J X X X X X X X
    # K X X X X X X X

    for k in range(len(caracteres_obtidos)):
        for i in range(len(labels)):
            if caracteres_obtidos[k] == labels[i]:
                adicionar_i = i
            if caracteres_esperados[k] == labels[i]:
                adicionar_j = i

        if caracteres_obtidos[k] != "None":
            matrizConfusao[adicionar_i][adicionar_j] += 1

        adicionar_i = 0
        adicionar_j = 0

    return matrizConfusao
