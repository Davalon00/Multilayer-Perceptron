import csv
import numpy as np
from mlp import MLP
from saidas import salvar_resultadoPorArquivo

# Função main --> final do arquivo.
# Inicialização de variáveis, não modificar.
taxa_aprendizado = 1
condicao_erro = 0.01
arquivoTeste = ""

def lerInputs(nomeArquivo):
    with open(nomeArquivo, encoding='utf-8-sig') as csvArquivo:
        readCSV = csv.reader(csvArquivo, delimiter=',')
        inputs = []
        i = 0

        for row in readCSV:
            entrada_inputs = []

            while i < 70:
                if i < 63:
                    entrada_inputs.append(int(row[i]))
                i += 1
            i = 0

            inputs.append(entrada_inputs)

    csvArquivo.close()
    return inputs


def lerLabels(nomeArquivo):
    with open(nomeArquivo, encoding='utf-8-sig') as csvArquivo:
        readCSV = csv.reader(csvArquivo, delimiter=',')
        labels = []
        i = 0

        for row in readCSV:

            entrada_labels = []
            while i < 70:
                if 62 < i < 70:
                    entrada_labels.append(int(row[i]))
                i += 1
            i = 0

            labels.append(entrada_labels)

    csvArquivo.close()
    return labels


def treinarSemValidacao(mlp, inputs_treino, labels_treino, inputs_teste, labels_teste):

    # treino
    mlp.treinoSemValidacao(np.array(inputs_treino), np.array(labels_treino), taxa_aprendizado, condicao_erro)
    # teste
    outputFinal = mlp.forwardPropagation(inputs_teste)

    printResult(inputs_teste, labels_teste, outputFinal)
    salvar_resultadoPorArquivo(outputFinal, labels_teste, arquivoTeste)


def treinarComValidacao(mlp, inputs_treino, labels_treino, inputs_teste, labels_teste, inputs_validacao, labels_validacao):

    # treino com validação
    mlp.treinoComValidacaoGrafico(np.array(inputs_treino), np.array(inputs_validacao), np.array(labels_treino),
                                  np.array(labels_validacao), taxa_aprendizado, condicao_erro)

    # teste
    outputFinal = mlp.forwardPropagation(inputs_teste)

    printResult(inputs_teste, labels_teste, outputFinal)
    salvar_resultadoPorArquivo(outputFinal, labels_teste, arquivoTeste)


def printResult(input, outputEsperado, outputObtido):
    indiceJ = 0
    indiceI = 0
    for i, letra in enumerate(input):  # para cada letra
        for j in range(9):  # para cada linha de 8 caracteres
            for k in range(7):  # para cada caractere
                if input[indiceI][indiceJ] == 1:
                    print(" 0 ", end="")
                if input[indiceI][indiceJ] == -1:
                    print("   ", end="")
                indiceJ += 1
            print("")
        print("")

        print("resultado esperado:")
        for k in range(7):
            print(outputEsperado[i][k], end=" ")
        print("")

        print("resultado obtido:")
        for k in range(7):
            print("{:.2f}".format(outputObtido[i][k]), end=" ")
        print("")
        print("")
        indiceJ = 0
        indiceI += 1

    print("")


if __name__ == '__main__':

    # Utiliza um desses arquivos.
    # "caracteres-limpo.csv"
    # "caracteres-ruido.csv"
    # "caracteres_ruido20.csv"

    # ------------------ Para testar/treinar com diferentes arquivos, modificar os nomes abaixo ----------------- #
    arquivoTreino = "caracteres_ruido20.csv"
    arquivoValidacao = "caracteres-ruido.csv"         # Opcional caso treino seja sem validação
    arquivoTeste = "caracteres-limpo.csv"

    # ----------------- Salvando inputs e labels dos arquivos para utilizar na mlp ------------------ #

    inputs_treino = lerInputs(arquivoTreino)
    labels_treino = lerLabels(arquivoTreino)

    inputs_teste = lerInputs(arquivoTeste)
    labels_teste = lerLabels(arquivoTeste)

    inputs_validacao = lerInputs(arquivoValidacao)
    labels_validacao = lerLabels(arquivoValidacao)

    # ---------------------------------- Criando a arquitetura da MLP ------------------------------- #

    mlp = MLP(63, [15], 7)          # 63 neurônios de entrada, 1 camada escondida com 15 neurônios, 7 neurônios de saída
    taxa_aprendizado = 0.1            # É possível alterar a taxa de aprendizado usada na mlp
    condicao_erro = 0.0001           # É possível alterar a condição de parada para a rede neural, em termos de erro.

    ########### OBS: AO UTILIZAR UM DOS TREINOS, COMENTAR O OUTRO, PARA NÃO CONFLITAR  ################
    # -------------------------------------- Treino sem validação  ---------------------------------- #

    # 1ª Análise - Treinando com o arquivo contendo ruidos e testendo com os outros, quantas épocas levam até chegar em 0.01?
    # 2ª Análise - Treinando com o arquivo limpo e testando com os outros, quantas épocas levam até checar em 0.01?
    # Apresentar saídas (incluindo matriz de confusão)

    # treinarSemValidacao(mlp, inputs_treino, labels_treino, inputs_teste, labels_teste)

    # -------------------------------------- Treino com validação  ---------------------------------- #

    # treinarComValidacao(mlp, inputs_treino, labels_treino, inputs_teste, labels_teste, inputs_validacao, labels_validacao)
