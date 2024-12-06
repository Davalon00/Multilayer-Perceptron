import numpy as np
from saidas import salvar_pesosIniciais, salvar_pesosFinais, salvar_parametros, salvar_erroPorEpoca, plotar_erroPorEpoca


class MLP(object):
    def __init__(self, numNeuronioEntrada, camadasEscondidas, numNeuronioSaida):
        self.numNeuronioEntrada = numNeuronioEntrada                            #variável que guarda o número de entradas/número de neurônios da camada inicial.
        self.camadasEscondidas = camadasEscondidas                              #variável que guarda uma lista, onde o índice é o nº da camada escondida, e o valor dentro é a quantidade de neurônios dessa camada escondida.
        self.numNeuronioSaida = numNeuronioSaida                                #variável que guarda o número de saidas/número de neurônios da camada final.
        self.lista_erros = []
        self.lista_validacao = []

        layers = [numNeuronioEntrada] + camadasEscondidas + [numNeuronioSaida]  #criação de uma lista que guarda o número de cada camada, juntando a informação das 3 variáveis acima.

        pesos = []                                                              #variável que guarda os pesos em uma matriz para cada camada (é um array tridimensional).
        for i in range(len(layers) - 1):                                        #para cada camada, gera uma matriz com valores aleatórios entre -1 e 1 e a adiciona na variável pesos.
            pesosAux = np.random.uniform(-1, 1, (layers[i], layers[i+1]))
            pesos.append(pesosAux)
        self.pesos = pesos

        salvar_pesosIniciais(self.pesos)                                        #salva o valor inicial dos pesos em um txt

        derivadas = []                                                          #variável que guarda as deviradas em uma matriz para cada camada (é um array tridimensional).
        for i in range(len(layers) - 1):                                        #para cada camada, gera uma matriz preenchida por zeros e a adiciona na variável derivadas.
            derivadasAux = np.zeros((layers[i], layers[i + 1]))
            derivadas.append(derivadasAux)
        self.derivadas = derivadas

        ativacoes = []                                                          #variável que guarda os valores de ativação para cada camada (é uma matriz).
        for i in range(len(layers)):                                            #para cada camada, gera um array preenchido por zeros e o adiciona na variável ativacoes.
            ativacoesAux = np.zeros(layers[i])
            ativacoes.append(ativacoesAux)
        self.ativacoes = ativacoes

    def forwardPropagation(self, entradas):

        ativacoes = entradas                                                    #os valores iniciais de ativação são os próprios entradas
        self.ativacoes[0] = ativacoes                                           #salva esses valores das ativações iniciais para a realização do back propagation futuramente.

        for i, peso in enumerate(self.pesos):                                   #para cada "camada de pesos"
            matrizEntradas = np.dot(ativacoes, peso)                            #realiza a multiplicação de matrizes entre a ativação atual e os pesos, gerando a matriz com os valores da ativação do camada atual.
            ativacoes = self.sigmoid(matrizEntradas)                            #aplica a função sigmoid aos resultados obtidos na multiplicação de matrizes executada na linha acima
            self.ativacoes[i + 1] = ativacoes                                   #salva o resultado das ativações obtidas na camada seguinte

        return ativacoes                                                        #retorna o último camada de ativações, ou seja, os resultados obtidos pelo algoritmo.

    def backPropagation(self, error):

        for i in reversed(range(len(self.derivadas))):                          #para cada camada de neurônios, indo do índice mais alto para o mais baixo (por isso o uso do reversed)
            ativacoes = self.ativacoes[i+1]                                     #pega a camada anterior de ativações (lembre-se que é o anterior considerando que estamos utilizando índices invertidos)
            delta = error * self.sigmoidDerivada(ativacoes)                     #calcula o delta (um array com o número de neurônios da camada anterior), que é o coeficiente de erro multiplicado pela devirada da função sigmoid aplicada nos valores de ativação
            delta_re = delta.reshape(delta.shape[0], -1).T                      #transforma o delta em uma matriz 1xN, pois utilizaremos ela em uma operação do numpy posteriormente, tornando necessário a transformação de array para matriz. Não há alteração nos valores armazenados
            current_ativacoes = self.ativacoes[i]                               #pega a camada atual de ativações, que é um array
            current_ativacoes = current_ativacoes.reshape(current_ativacoes.shape[0],-1)  #transforma o array de ativações em uma matriz Nx1, pois utilizaremos ela em uma opreção do numpy posteriormente, tornando necessário a transformação de array para matriz. Não há alteração nos valores armazenados
            self.derivadas[i] = np.dot(current_ativacoes, delta_re)             #realiza a multiplicação de matrizes entre a camada atual de ativações e o delta para calcular o valor das derivadas da camada atual, resultado que será utilizado no gradiente posteriormente (em outras palavras, pela o valor de ativação de um neurônio e multiplica pelo delta do mesmo neurônio, gerando uma matriz NxM, onde N é o número de neurônios do layer atual, e M o número de neurônios do layer anterior)
            error = np.dot(delta, self.pesos[i].T)                              #realiza a multiplicação entre o delta e os pesos entre os neurônios da camada atual e da camada anterior (sendo que a matriz das camadas é transposta, pois queremos gerar uma matriz 1xN, onde o N é o número de neurônios da camada anterior, o que não seria possível sem transpor a matriz, pois estamos na ordem reversed), obtendo os valores dos erros por neurônio (que é a somatória do delta x peso N).

    def treinoComValidacao(self, entradas, validacao, respostas, respostasValidacao, taxaAprendizado, erroCondicao):                  # entradas e labels: array (numpy) | taxa_aprendizado: float!

        salvar_parametros(self.numNeuronioEntrada, self.camadasEscondidas[0], self.numNeuronioSaida, taxaAprendizado, erroCondicao) #printa em um arquivo de saída os números de neurônios das camadas, a taxa de aprendizado e a condição de parada.

        erroEpoca = erroCondicao + 1  # Apenas para entrar no loop inicialmente
        erroValidacaoAntigo = 999999

        i = 1   #iterador

        while erroEpoca > erroCondicao:                                         #enquanto o nosso programa estiver errando mais do que a qualidade desejada, continua treinando
            somatoriaErros = 0                                                  #variável que guarda a somatória dos erros
            errosValidacaoEpoca = 0

            for j, entrada in enumerate(entradas):                              #para cada letra da entrada
                resposta = respostas[j]                                         #seleciona o label dessa entrada
                saidaTreino = self.forwardPropagation(entrada)                  #roda o programa para obter os resultados
                erroEntrada = resposta - saidaTreino                            #calcula o erro obtido pelo programa, que é um array com as diferenças de cada neurônio de saída.
                self.backPropagation(erroEntrada)                               #executa o back propagate
                self.executaGradiente(taxaAprendizado)                          #executa o gradiente para alterar os pesos/aprender
                somatoriaErros += self.calcErroMedio(resposta, saidaTreino)     #soma o erro do teste dessa letra a variável que possui a soma do erro de todas as letras

                respostaValidacao = respostasValidacao[j]                       #seleciona o label da entrada de validacao
                saidaValidacao = self.forwardPropagation(validacao)             #realiza o forward propagate com a entrada de validacao
                errosValidacaoEpoca += self.calcErroMedio(respostaValidacao, saidaValidacao)#calcula a média de erro da camada de validacao e soma a variável que realiza a somatória desses erros


            erroEpoca = somatoriaErros/self.numNeuronioEntrada                  #calcula o erro por neurônio
            salvar_erroPorEpoca(erroEpoca, i)                                   #salva o resultado dessa época em um arquivo txt
            self.lista_erros.append(erroEpoca)

            if errosValidacaoEpoca > erroValidacaoAntigo:                       #se o erro da execucao da validacao dessa época for um erro maior do que o da camada anterior, significa que o fenomeno do overfitting está ocorrendo, logo, devemos parar o treinamento
                print("a IA entrou em processo de overfitting, executando parada antecipada na epoca: ", end="")
                print(i, end="")
                print(" - erro desta época: ", end="")
                print(errosValidacaoEpoca, end="")
                print(" - erro da época passada: ", end="")
                print(erroValidacaoAntigo, end="")
                print(" - diferença entre eles: ", end="")
                print(errosValidacaoEpoca-erroValidacaoAntigo)
                break
            else:                                                               #caso o fenômeno não esteja ocorrendo, salvamos o erro da época atual para compararmos na próxima época
                erroValidacaoAntigo = errosValidacaoEpoca

            i += 1                                                              #adiciona 1 no índice de épocas

        salvar_pesosFinais(self.pesos)                                          #salva quais são os pesos finais em um arquivo txt
        plotar_erroPorEpoca(self.lista_erros)                                   #plota os erros por época em um gráfico png

    def treinoComValidacaoGrafico(self, entradas, validacao, respostas, respostasValidacao, taxaAprendizado, erroCondicao):                  # entradas e labels: array (numpy) | taxa_aprendizado: float!

        salvar_parametros(self.numNeuronioEntrada, self.camadasEscondidas[0], self.numNeuronioSaida, taxaAprendizado, erroCondicao) #printa em um arquivo de saída os números de neurônios das camadas, a taxa de aprendizado e a condição de parada.

        erroEpoca = erroCondicao + 1  # Apenas para entrar no loop inicialmente

        i = 1   #iterador

        while erroEpoca > erroCondicao:                                         #enquanto o nosso programa estiver errando mais do que a qualidade desejada, continua treinando
            somatoriaErros = 0                                                  #variável que guarda a somatória dos erros
            errosValidacaoEpoca = 0

            for j, entrada in enumerate(entradas):                              #para cada letra da entrada
                resposta = respostas[j]                                         #seleciona o label dessa entrada
                saidaTreino = self.forwardPropagation(entrada)                  #roda o programa para obter os resultados
                erroEntrada = resposta - saidaTreino                            #calcula o erro obtido pelo programa, que é um array com as diferenças de cada neurônio de saída.
                self.backPropagation(erroEntrada)                               #executa o back propagate
                self.executaGradiente(taxaAprendizado)                          #executa o gradiente para alterar os pesos/aprender
                somatoriaErros += self.calcErroMedio(resposta, saidaTreino)     #soma o erro do teste dessa letra a variável que possui a soma do erro de todas as letras

                respostaValidacao = respostasValidacao[j]                       #seleciona o label da entrada de validacao
                saidaValidacao = self.forwardPropagation(validacao)             #realiza o forward propagate com a entrada de validacao
                errosValidacaoEpoca += self.calcErroMedio(respostaValidacao, saidaValidacao)#calcula a média de erro da camada de validacao e soma a variável que realiza a somatória desses erros


            erroEpoca = somatoriaErros/self.numNeuronioEntrada                  #calcula o erro por neurônio
            errosValidacaoEpoca = errosValidacaoEpoca/self.numNeuronioEntrada   #calcula o erro por neurônio do teste de validacao
            salvar_erroPorEpoca(erroEpoca, i)                                   #salva o resultado dessa época em um arquivo txt
            self.lista_erros.append(erroEpoca)
            self.lista_validacao.append(errosValidacaoEpoca)

            i += 1                                                              #adiciona 1 no índice de épocas

        salvar_pesosFinais(self.pesos)                                          #salva quais são os pesos finais em um arquivo txt
        plotar_erroPorEpoca(self.lista_erros)                                   #plota os erros por época em um gráfico png
        plotar_erroPorEpoca(self.lista_validacao)                               #plota os erros por época dos testes de validacao

    def treinoSemValidacao(self, entradas, respostas, taxaAprendizado, erroCondicao):                  # entradas e labels: array (numpy) | taxa_aprendizado: float!

        salvar_parametros(self.numNeuronioEntrada, self.camadasEscondidas[0], self.numNeuronioSaida, taxaAprendizado, erroCondicao) #printa em um arquivo de saída os números de neurônios das camadas, a taxa de aprendizado e a condição de parada.

        erroEpoca = erroCondicao + 1  # Apenas para entrar no loop inicialmente

        i = 1   #iterador

        while erroEpoca > erroCondicao:                                         #enquanto o nosso programa estiver errando mais do que a qualidade desejada, continua treinando
            somatoriaErros = 0                                                  #variável que guarda a somatória dos erros

            for j, entrada in enumerate(entradas):                              #para cada letra da entrada
                resposta = respostas[j]                                         #seleciona o label dessa entrada
                saidaTreino = self.forwardPropagation(entrada)                  #roda o programa para obter os resultados
                erroEntrada = resposta - saidaTreino                            #calcula o erro obtido pelo programa, que é um array com as diferenças de cada neurônio de saída.
                self.backPropagation(erroEntrada)                               #executa o back propagate
                self.executaGradiente(taxaAprendizado)                          #executa o gradiente para alterar os pesos/aprender
                somatoriaErros += self.calcErroMedio(resposta, saidaTreino)     #soma o erro do teste dessa letra a variável que possui a soma do erro de todas as letras

            erroEpoca = somatoriaErros/self.numNeuronioEntrada                  #calcula o erro por neurônio
            salvar_erroPorEpoca(erroEpoca, i)                                   #salva o resultado dessa época em um arquivo txt
            self.lista_erros.append(erroEpoca)

            i += 1                                                              #adiciona 1 no índice de épocas

        salvar_pesosFinais(self.pesos)                                          #salva quais são os pesos finais em um arquivo txt
        plotar_erroPorEpoca(self.lista_erros)                                   #plota os erros por época em um gráfico png

    def executaGradiente(self, taxaAprendizado):                                # Entrada: Float | taxa de aprendizado = Quão rápido aprende
        for i in range(len(self.pesos)):                                        #para cada camada de pesos
            pesos = self.pesos[i]                                               #pega esse camada
            derivadas = self.derivadas[i]                                       #pega as derivadas desse camada
            pesos += derivadas * taxaAprendizado                                #soma um valor a cada peso, que é o valor da derivada multiplicado pelo coeficiente de aprendizado

    def sigmoid(self, x):                                                       # Entrada: Float, Retorna: Float
        return 1.0 / (1 + np.exp(-x))                                           #calcula a função sigmoid

    def sigmoidDerivada(self, x):                                               # Entrada: Float, Retorna: Float
        return x * (1.0 - x)                                                    #calcula a derivada da função sigmoid

    def calcErroMedio(self, resposta, saida):                                            # Entrada: array (numpy), Retorna Float
        return np.average((resposta - saida) ** 2)                              #calcula uma média dos erros, sendo que ela é modular