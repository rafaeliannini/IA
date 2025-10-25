import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid', learning_rate=0.1):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation_name = activation
        
        # Inicialização dos pesos com Xavier/He initialization
        if activation == 'relu':
            # He initialization para ReLU
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
            self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        else:
            # Xavier initialization para sigmoid/tanh
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
            self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        
        # Inicialização dos biases
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))
        
        # Histórico de treinamento
        self.loss_history = []
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def tanh(self, z):
        return np.tanh(z)
    
    def tanh_derivative(self, z):
        return 1 - np.tanh(z) ** 2
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def activation(self, z):
        if self.activation_name == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation_name == 'relu':
            return self.relu(z)
        elif self.activation_name == 'tanh':
            return self.tanh(z)
        else:
            return self.sigmoid(z)
    
    def activation_derivative(self, z):
        if self.activation_name == 'sigmoid':
            return self.sigmoid_derivative(z)
        elif self.activation_name == 'relu':
            return self.relu_derivative(z)
        elif self.activation_name == 'tanh':
            return self.tanh_derivative(z)
        else:
            return self.sigmoid_derivative(z)
    
    def forward(self, X):

        # Camada oculta
        self.Z1 = np.dot(X, self.W1) + self.b1  # Combinação linear
        self.A1 = self.activation(self.Z1)       # Ativação
        
        # Camada de saída
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # Combinação linear
        self.A2 = self.sigmoid(self.Z2)                # Ativação
        
        return self.A2
    
    def backward(self, X, y, output):
        
        m = X.shape[0]  # número de amostras
        
        # Erro na camada de saída (δ2)
        # Para MSE com sigmoid: δ2 = (ŷ - y) * σ'(Z2)
        dZ2 = (output - y) * self.sigmoid_derivative(self.Z2)
        
        # Gradientes da camada de saída
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Erro propagado para camada oculta (δ1)
        dZ1 = np.dot(dZ2, self.W2.T) * self.activation_derivative(self.Z1)
        
        # Gradientes da camada oculta
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Atualização dos pesos
        self.W1 -= self.learning_rate * dW1
        self.W2 -= self.learning_rate * dW2
        self.b1 -= self.learning_rate * db1
        self.b2 -= self.learning_rate * db2
    
    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = (1/(2*m)) * np.sum((y_pred - y_true) ** 2)
        return loss
    
    def train(self, X, y, epochs=1000, verbose=True):

        for epoch in range(epochs):
            # Forward propagation
            output = self.forward(X)
            
            # Backward propagation
            self.backward(X, y, output)
            
            # Calcula e armazena a loss
            loss = self.compute_loss(y, output)
            self.loss_history.append(loss)
    
    def predict(self, X):
        return self.forward(X)

# PROBLEMA 1: XOR

print("=" * 80)
print("PROBLEMA 1: XOR")
print("=" * 80)

# Dados de treinamento para XOR
X_xor = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

y_xor = np.array([[0],
                  [1],
                  [1],
                  [0]])

print("\nTabela verdade XOR:")
print("X1  X2  |  Y")
print("-" * 15)
for i in range(len(X_xor)):
    print(f"{X_xor[i][0]}   {X_xor[i][1]}   |  {y_xor[i][0]}")

# Treinar rede para XOR com diferentes funções de ativação
activations = ['sigmoid', 'tanh', 'relu']
results_xor = {}

for activation in activations:
    print(f"\n{'='*60}")
    print(f"Treinando com função de ativação: {activation.upper()}")
    print(f"{'='*60}")
    
    # Criar e treinar a rede
    nn_xor = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, 
                           activation=activation, learning_rate=0.5)
    
    nn_xor.train(X_xor, y_xor, epochs=5000, verbose=True)
    
    # Fazer predições
    predictions = nn_xor.predict(X_xor)
    
    print(f"\nResultados após o treinamento ({activation}):")
    print("Entrada  |  Esperado  |  Predito  |  Arredondado")
    print("-" * 50)
    for i in range(len(X_xor)):
        pred_rounded = 1 if predictions[i][0] > 0.5 else 0
        print(f"{X_xor[i]}  |     {y_xor[i][0]}      |   {predictions[i][0]:.4f}  |      {pred_rounded}")
    
    results_xor[activation] = nn_xor

# Plotar as curvas de aprendizado
plt.figure(figsize=(12, 4))
for i, (activation, nn) in enumerate(results_xor.items()):
    plt.subplot(1, 3, i+1)
    plt.plot(nn.loss_history)
    plt.title(f'Curva de Aprendizado - {activation.upper()}')
    plt.xlabel('Época')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
plt.tight_layout()
plt.savefig('xor_learning_curves.png', dpi=300, bbox_inches='tight')
print("\nGráfico salvo como 'xor_learning_curves.png'")

# PROBLEMA 2: DISPLAY DE 7 SEGMENTOS

print("\n" + "=" * 80)
print("PROBLEMA 2: DISPLAY DE 7 SEGMENTOS")
print("=" * 80)

# Dados de treinamento (segmentos a até g)
X_segments = np.array([
    [1, 1, 1, 1, 1, 1, 0],  # 0
    [0, 1, 1, 0, 0, 0, 0],  # 1
    [1, 1, 0, 1, 1, 0, 1],  # 2
    [1, 1, 1, 1, 0, 0, 1],  # 3
    [0, 1, 1, 0, 0, 1, 1],  # 4
    [1, 0, 1, 1, 0, 1, 1],  # 5
    [1, 0, 1, 1, 1, 1, 1],  # 6
    [1, 1, 1, 0, 0, 0, 0],  # 7
    [1, 1, 1, 1, 1, 1, 1],  # 8
    [1, 1, 1, 1, 0, 1, 1],  # 9
])

# Saída esperada (one-hot encoding)
y_segments = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 2
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 3
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 4
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 5
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 6
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 7
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 9
])

print("\nTabela de segmentos:")
print("Dígito | a b c d e f g | One-hot")
print("-" * 50)
for i in range(len(X_segments)):
    segments_str = ' '.join(map(str, X_segments[i]))
    onehot_idx = np.argmax(y_segments[i])
    print(f"  {i}    | {segments_str} | dígito {onehot_idx}")

# Treinar rede para display de 7 segmentos com diferentes funções de ativação
results_segments = {}

for activation in activations:
    print(f"\n{'='*60}")
    print(f"Treinando com função de ativação: {activation.upper()}")
    print(f"{'='*60}")
    
    # Criar e treinar a rede
    nn_segments = NeuralNetwork(input_size=7, hidden_size=5, output_size=10, 
                                activation=activation, learning_rate=0.5)
    
    nn_segments.train(X_segments, y_segments, epochs=10000, verbose=True)
    
    # Fazer predições
    predictions = nn_segments.predict(X_segments)
    
    print(f"\nResultados após o treinamento ({activation}):")
    print("Dígito Real | Dígito Predito | Confiança | Correto?")
    print("-" * 60)
    
    correct = 0
    for i in range(len(X_segments)):
        real_digit = i
        predicted_digit = np.argmax(predictions[i])
        confidence = predictions[i][predicted_digit]
        is_correct = "✓" if real_digit == predicted_digit else "✗"
        
        if real_digit == predicted_digit:
            correct += 1
        
        print(f"     {real_digit}      |       {predicted_digit}        |   {confidence:.4f}  |    {is_correct}")
    
    accuracy = (correct / len(X_segments)) * 100
    print(f"\nAcurácia: {accuracy:.2f}%")
    
    results_segments[activation] = nn_segments

# Plotar as curvas de aprendizado
plt.figure(figsize=(12, 4))
for i, (activation, nn) in enumerate(results_segments.items()):
    plt.subplot(1, 3, i+1)
    plt.plot(nn.loss_history)
    plt.title(f'Curva de Aprendizado - {activation.upper()}')
    plt.xlabel('Época')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
plt.tight_layout()
plt.savefig('segments_learning_curves.png', dpi=300, bbox_inches='tight')
print("\nGráfico salvo como 'segments_learning_curves.png'")

# TESTE DE ROBUSTEZ COM RUÍDO

print("\n" + "=" * 80)
print("TESTE DE ROBUSTEZ - ADICIONANDO RUÍDO AOS SEGMENTOS")
print("=" * 80)

def add_noise_to_segments(X, noise_type='flip', noise_level=1):

    X_noisy = X.copy()
    n_samples, n_features = X.shape
    
    for i in range(n_samples):
        # Seleciona aleatoriamente quais segmentos serão afetados
        segments_to_modify = np.random.choice(n_features, size=noise_level, replace=False)
        
        for seg in segments_to_modify:
            if noise_type == 'flip':
                # Inverte o bit (0→1 ou 1→0)
                X_noisy[i, seg] = 1 - X_noisy[i, seg]
            elif noise_type == 'off':
                # Desliga o segmento (força para 0)
                X_noisy[i, seg] = 0
            elif noise_type == 'on':
                # Liga o segmento (força para 1)
                X_noisy[i, seg] = 1
    
    return X_noisy


# Testar cada modelo treinado com diferentes tipos e níveis de ruído
noise_types = ['flip', 'off', 'on']
noise_levels = [1, 2, 3]  # número de segmentos afetados

print("\nTestando robustez dos modelos...")
print("=" * 80)

# Dicionário para armazenar resultados
robustness_results = {}

for activation in activations:
    print(f"\n{'='*70}")
    print(f"Modelo: {activation.upper()}")
    print(f"{'='*70}")
    
    nn = results_segments[activation]
    robustness_results[activation] = {}
    
    for noise_type in noise_types:
        print(f"\n--- Tipo de ruído: {noise_type.upper()} ---")
        robustness_results[activation][noise_type] = {}
        
        for noise_level in noise_levels:
            # Gerar dados com ruído
            X_noisy = add_noise_to_segments(X_segments, noise_type, noise_level)
            
            # Fazer predições
            predictions_noisy = nn.predict(X_noisy)
            
            # Calcular acurácia
            correct = 0
            for i in range(len(X_noisy)):
                real_digit = i
                predicted_digit = np.argmax(predictions_noisy[i])
                if real_digit == predicted_digit:
                    correct += 1
            
            accuracy = (correct / len(X_noisy)) * 100
            robustness_results[activation][noise_type][noise_level] = accuracy
            
            print(f"  Ruído nível {noise_level} ({noise_level} segmento(s) afetado(s)): "
                  f"Acurácia = {accuracy:.2f}%")

print("\n" + "=" * 80)
print("EXEMPLOS DETALHADOS - COMPARAÇÃO COM E SEM RUÍDO")
print("=" * 80)

best_activation = 'sigmoid'
nn_best = results_segments[best_activation]

print(f"\nUsando modelo: {best_activation.upper()}")
print("\nComparação: Original vs. Com Ruído (1 segmento invertido)")
print("=" * 90)

# Gerar uma amostra com ruído para cada dígito
np.random.seed(42)
X_noisy_demo = add_noise_to_segments(X_segments, 'flip', 1)

print("\nDígito | Segmentos Originais | Segmentos com Ruído | Pred. Original | Pred. Ruído | Status")
print("-" * 90)

for i in range(len(X_segments)):
    orig_segments = ''.join(map(str, X_segments[i]))
    noisy_segments = ''.join(map(str, X_noisy_demo[i]))
    
    # Predição original
    pred_orig = nn_best.predict(X_segments[i:i+1])
    digit_orig = np.argmax(pred_orig)
    conf_orig = pred_orig[0][digit_orig]
    
    # Predição com ruído
    pred_noisy = nn_best.predict(X_noisy_demo[i:i+1])
    digit_noisy = np.argmax(pred_noisy)
    conf_noisy = pred_noisy[0][digit_noisy]
    
    # Status
    if digit_orig == i and digit_noisy == i:
        status = "✓✓ (Ambos corretos)"
    elif digit_orig == i and digit_noisy != i:
        status = f"✗ (Erro com ruído: predisse {digit_noisy})"
    else:
        status = "✗✗ (Erro sem ruído)"
    
    print(f"  {i}    |    {orig_segments}    |      {noisy_segments}      |"
          f"   {digit_orig} ({conf_orig:.3f})   |   {digit_noisy} ({conf_noisy:.3f})  | {status}")


# GRÁFICO DE ROBUSTEZ

print("\n" + "=" * 80)
print("GERANDO GRÁFICO DE ROBUSTEZ")
print("=" * 80)

# Criar gráfico de robustez para cada tipo de ruído
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, noise_type in enumerate(noise_types):
    ax = axes[idx]
    
    for activation in activations:
        accuracies = [robustness_results[activation][noise_type][level] 
                     for level in noise_levels]
        ax.plot(noise_levels, accuracies, marker='o', label=activation.upper(), linewidth=2)
    
    ax.set_xlabel('Nível de Ruído (segmentos afetados)', fontsize=11)
    ax.set_ylabel('Acurácia (%)', fontsize=11)
    ax.set_title(f'Robustez - Ruído tipo "{noise_type.upper()}"', fontsize=12, fontweight='bold')
    ax.set_xticks(noise_levels)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Adicionar linha de referência em 100%
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.3, label='100%')

plt.tight_layout()
plt.savefig('segments_robustness.png', dpi=300, bbox_inches='tight')
print("\nGráfico de robustez salvo como 'segments_robustness.png'")


# ANÁLISE DE SENSIBILIDADE POR SEGMENTO

print("\n" + "=" * 80)
print("ANÁLISE DE SENSIBILIDADE - QUAL SEGMENTO É MAIS CRÍTICO?")
print("=" * 80)

nn_analysis = results_segments['sigmoid']

segment_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
segment_sensitivity = {seg: [] for seg in segment_names}

print("\nTestando impacto da falha de cada segmento individual...")

for seg_idx, seg_name in enumerate(segment_names):
    print(f"\nAnalisando segmento '{seg_name}'...")
    
    # Para cada dígito, desligar apenas esse segmento
    correct_count = 0
    
    for digit_idx in range(len(X_segments)):
        X_modified = X_segments[digit_idx:digit_idx+1].copy()
        
        # Desligar o segmento específico
        original_value = X_modified[0, seg_idx]
        X_modified[0, seg_idx] = 0
        
        # Fazer predição
        pred = nn_analysis.predict(X_modified)
        predicted_digit = np.argmax(pred)
        
        # Verificar se acertou
        if predicted_digit == digit_idx:
            correct_count += 1
    
    accuracy = (correct_count / len(X_segments)) * 100
    segment_sensitivity[seg_name] = accuracy
    print(f"  Segmento '{seg_name}' desligado → Acurácia: {accuracy:.1f}%")

# Plotar sensibilidade por segmento
plt.figure(figsize=(10, 6))
segments = list(segment_sensitivity.keys())
accuracies = list(segment_sensitivity.values())

colors = ['red' if acc < 70 else 'orange' if acc < 85 else 'green' for acc in accuracies]
bars = plt.bar(segments, accuracies, color=colors, alpha=0.7, edgecolor='black')

plt.xlabel('Segmento', fontsize=12)
plt.ylabel('Acurácia quando segmento falha (%)', fontsize=12)
plt.title('Sensibilidade: Impacto da Falha de Cada Segmento', fontsize=14, fontweight='bold')
plt.ylim([0, 105])
plt.axhline(y=100, color='green', linestyle='--', alpha=0.3, linewidth=2, label='100% (sem falha)')
plt.grid(True, alpha=0.3, axis='y')

# Adicionar valores nas barras
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.legend()
plt.tight_layout()
plt.savefig('segment_sensitivity.png', dpi=300, bbox_inches='tight')
print("\nGráfico de sensibilidade salvo como 'segment_sensitivity.png'")

print("\n" + "=" * 80)
print("SUMÁRIO FINAL DOS TESTES DE ROBUSTEZ")
print("=" * 80)

print("\n1. ACURÁCIA SEM RUÍDO:")
for activation in activations:
    nn = results_segments[activation]
    preds = nn.predict(X_segments)
    correct = sum([1 for i in range(len(X_segments)) if np.argmax(preds[i]) == i])
    acc = (correct / len(X_segments)) * 100
    print(f"   {activation.upper():8s}: {acc:.2f}%")

print("\n2. ROBUSTEZ MÉDIA (todos os tipos de ruído, nível 1):")
for activation in activations:
    avg_robustness = np.mean([robustness_results[activation][nt][1] for nt in noise_types])
    print(f"   {activation.upper():8s}: {avg_robustness:.2f}%")

print("\n3. SEGMENTOS MAIS CRÍTICOS (quando falham):")
sorted_segments = sorted(segment_sensitivity.items(), key=lambda x: x[1])
print("   Segmentos com maior impacto negativo quando falham:")
for seg, acc in sorted_segments[:3]:
    impact = 100 - acc
    print(f"   - Segmento '{seg}': {impact:.1f}% de degradação")

print("\n" + "=" * 80)
print("Análise completa de robustez concluída!")
print("Verifique os gráficos gerados:")
print("  - segments_learning_curves.png")
print("  - segments_robustness.png")
print("  - segment_sensitivity.png")
print("=" * 80)

# DEMONSTRAÇÃO DAS EQUAÇÕES DE BACKPROPAGATION

print("\n" + "=" * 80)
print("DEMONSTRAÇÃO DAS EQUAÇÕES DE BACKPROPAGATION")
print("=" * 80)

print("""
EQUAÇÕES DE AJUSTE DOS PESOS DURANTE O TREINAMENTO:

1. FORWARD PROPAGATION (Propagação para frente):
   
   Z1 = X · W1 + b1                    (Combinação linear - camada oculta)
   A1 = σ(Z1)                          (Ativação - camada oculta)
   Z2 = A1 · W2 + b2                   (Combinação linear - camada de saída)
   Ŷ = A2 = σ(Z2)                      (Ativação - camada de saída)

2. CÁLCULO DO ERRO (Loss):
   
   E = (1/n) Σ(yi - ŷi)²               (Mean Squared Error - MSE)

3. BACKWARD PROPAGATION (Propagação para trás):
   
   a) Erro na camada de saída:
      δ2 = (Ŷ - Y) ⊙ σ'(Z2)            (Delta da camada de saída)
      
   b) Gradientes da camada de saída:
      ∂E/∂W2 = (1/m) · A1ᵀ · δ2        (Gradiente dos pesos W2)
      ∂E/∂b2 = (1/m) · Σ δ2            (Gradiente dos biases b2)
      
   c) Erro propagado para camada oculta:
      δ1 = (δ2 · W2ᵀ) ⊙ σ'(Z1)         (Delta da camada oculta)
      
   d) Gradientes da camada oculta:
      ∂E/∂W1 = (1/m) · Xᵀ · δ1         (Gradiente dos pesos W1)
      ∂E/∂b1 = (1/m) · Σ δ1            (Gradiente dos biases b1)

4. ATUALIZAÇÃO DOS PESOS (Gradient Descent):
   
   W1 ← W1 - α · ∂E/∂W1                (Atualiza W1)
   W2 ← W2 - α · ∂E/∂W2                (Atualiza W2)
   b1 ← b1 - α · ∂E/∂b1                (Atualiza b1)
   b2 ← b2 - α · ∂E/∂b2                (Atualiza b2)

Onde:
- ⊙ é o produto elemento a elemento (Hadamard product)
- α é a taxa de aprendizado (learning rate)
- σ é a função de ativação (sigmoid, ReLU, tanh)
- σ' é a derivada da função de ativação
- m é o número de amostras no batch
- Xᵀ, A1ᵀ, W2ᵀ são as matrizes transpostas

DERIVADAS DAS FUNÇÕES DE ATIVAÇÃO:

1. Sigmoid: σ(z) = 1/(1 + e^(-z))
   σ'(z) = σ(z) · (1 - σ(z))

2. ReLU: f(z) = max(0, z)
   f'(z) = 1 se z > 0, 0 caso contrário

3. Tanh: f(z) = tanh(z)
   f'(z) = 1 - tanh²(z)
""")

print("\n" + "=" * 80)
print("Treinamento concluído! Verifique os gráficos gerados.")
print("=" * 80)
