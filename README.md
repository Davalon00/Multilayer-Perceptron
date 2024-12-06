# Multi-Layer Perceptron (MLP) with Training and Validation
This repository contains two Python scripts for training and testing a **Multi-Layer Perceptron (MLP)** model with or without validation data. The MLP model uses forward propagation, backpropagation, and a sigmoid activation function for training. The code also supports plotting training and validation errors and saving the network's weights, errors, and parameters during training.

The three main scripts are:

1. **main.py**: Demonstrates how to train and test the MLP model on different datasets, including functions for reading data, training the model, and printing results.
2. **mlp.py**: Defines the `MLP` class, including methods for forward and backward propagation, gradient descent, and training with or without validation.
3. **saidas.py**: Provides functions for saving the model's weights, parameters, and errors to text files.
 
## Requirements

- **Python 3.x**
- **numpy**: Install using `pip install numpy`
- **matplotlib**: Install using `pip install matplotlib`

## MLP Class (`mlp.py`)

### Description
The `MLP` class provides a simple implementation of a feedforward neural network with multiple hidden layers. It includes methods for training the model, making predictions, and saving the results.

### Key Methods

- **`__init__(self, numNeuronioEntrada, camadasEscondidas, numNeuronioSaida)`**
  - Initializes the MLP model with input, hidden, and output layers.
  - Randomly initializes the weights and derivatives for backpropagation.
  
- **`forwardPropagation(self, entradas)`**
  - Performs forward propagation to calculate the activations for each layer.
  
- **`backPropagation(self, error)`**
  - Performs backpropagation to calculate the error gradients and update weights.
  
- **`treinoComValidacao(self, entradas, validacao, respostas, respostasValidacao, taxaAprendizado, erroCondicao)`**
  - Trains the model with validation data and stops early if overfitting is detected.

- **`treinoComValidacaoGrafico(self, entradas, validacao, respostas, respostasValidacao, taxaAprendizado, erroCondicao)`**
  - Similar to `treinoComValidacao`, but also generates plots of the training and validation errors.

- **`treinoSemValidacao(self, entradas, respostas, taxaAprendizado, erroCondicao)`**
  - Trains the model without validation data.

- **`executaGradiente(self, taxaAprendizado)`**
  - Performs gradient descent to update weights based on the calculated gradients and learning rate.

- **`sigmoid(self, x)`**
  - Sigmoid activation function.

- **`sigmoidDerivada(self, x)`**
  - Derivative of the sigmoid activation function.

- **`calcErroMedio(self, resposta, saida)`**
  - Calculates the mean squared error between the expected output and the actual output.

### Usage Example

```python
# Initialize the MLP model with input size 3, one hidden layer with 4 neurons, and 2 output neurons
mlp = MLP(3, [4], 2)

# Train the model with training data (inputs, labels) and test data (inputs_teste, labels_teste)
mlp.treinoComValidacao(inputs_treino, inputs_teste, labels_treino, labels_teste, taxaAprendizado=0.1, erroCondicao=0.01)
```

## Helper Functions (`saidas.py`)

### Description
The `saidas.py` file provides functions to save various model parameters, errors, and weights during training. These functions help log the training process and facilitate analysis.

### Key Functions

- **`salvar_pesosIniciais(pesos)`**
  - Saves the initial weights of the model to a text file.
  
- **`salvar_pesosFinais(pesos)`**
  - Saves the final weights after training to a text file.

- **`salvar_parametros(numNeuronioEntrada, camadasEscondidas, numNeuronioSaida, taxaAprendizado, erroCondicao)`**
  - Saves the model parameters (number of neurons in each layer, learning rate, and error condition) to a text file.
  
- **`salvar_erroPorEpoca(erro, epoca)`**
  - Saves the error for each epoch during training to a text file.
  
- **`plotar_erroPorEpoca(lista_erros)`**
  - Plots the errors over epochs using `matplotlib` and saves the plot as an image.

### Usage Example

```python
# Save the initial weights
salvar_pesosIniciais(mlp.pesos)

# Save the final weights after training
salvar_pesosFinais(mlp.pesos)

# Plot and save the error graph
plotar_erroPorEpoca(mlp.lista_erros)
```

## Training Script (`training_script.py`)

### Description
This script demonstrates how to train the MLP model using different datasets. It includes functions for reading input and output data, training the model with or without validation, and printing results.

### Key Functions

- **`lerInputs(nomeArquivo)`**
  - Reads input data from a CSV file and returns it as a NumPy array.
  
- **`lerLabels(nomeArquivo)`**
  - Reads the labels (outputs) from a CSV file and returns them as a NumPy array.
  
- **`treinarSemValidacao(mlp, inputs_treino, labels_treino, inputs_teste, labels_teste)`**
  - Trains the MLP model without validation data.
  
- **`treinarComValidacao(mlp, inputs_treino, labels_treino, inputs_teste, labels_teste, inputs_validacao, labels_validacao)`**
  - Trains the MLP model with validation data.

- **`printResult(input, outputEsperado, outputObtido)`**
  - Prints the expected and actual output for each test case.

### Usage Example

```python
# Read training data and labels
inputs_treino = lerInputs('dados_treino.csv')
labels_treino = lerLabels('labels_treino.csv')

# Read test data and labels
inputs_teste = lerInputs('dados_teste.csv')
labels_teste = lerLabels('labels_teste.csv')

# Train the MLP model with the training data
treinarComValidacao(mlp, inputs_treino, labels_treino, inputs_teste, labels_teste, inputs_validacao, labels_validacao)
```

## Dataset Files

- **dados_treino.csv**: Training dataset (input features).
- **labels_treino.csv**: Training labels (output classes).
- **dados_teste.csv**: Test dataset (input features).
- **labels_teste.csv**: Test labels (output classes).
- **dados_validacao.csv**: Validation dataset (input features).
- **labels_validacao.csv**: Validation labels (output classes).

## Results and Plotting

During training, the following results will be saved:

- **Initial weights**: Saved to a text file using `salvar_pesosIniciais`.
- **Final weights**: Saved after training using `salvar_pesosFinais`.
- **Errors per epoch**: Logged and saved using `salvar_erroPorEpoca` and plotted using `plotar_erroPorEpoca`.
- **Training parameters**: Saved in a text file using `salvar_parametros`.
