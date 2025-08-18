# Minha experiência com o tutorial de Transformer do TensorFlow

## O que entendi sobre o tutorial e sua importância

O tutorial de Transformer do TensorFlow me mostrou, de forma prática, como funciona a arquitetura que hoje é a base de modelos de linguagem modernos (como BERT e GPT). Eu percebi que a importância dele não está só em ensinar a programar um tradutor, mas sim em mostrar como o Transformer pensa, como ele organiza a informação e como isso o torna tão poderoso em tarefas de NLP.  

## Estrutura e O que entendi sobre cada etapa

1. **Preparação dos dados**  
   Percebi que o dataset não pode ser usado “cru”. Ele precisa passar por várias etapas:  
   - **Tokenização**: cada frase é quebrada em tokens (palavras ou subpalavras).  
   - **Adição de tokens especiais**: é necessário adicionar marcadores como `<start>` e `<end>` para indicar onde a frase começa e termina.  
   - **Padding**: frases mais curtas recebem preenchimento (zeros) para que todas fiquem do mesmo tamanho.  
   - **Truncamento**: frases muito longas são cortadas para não ultrapassar o limite definido.  
   Entendi que, sem esse preparo, o modelo não teria como alinhar as entradas com as saídas corretamente, e o treino não funcionaria.

2. **Embeddings posicionais**  
   O embedding posicional é o que dá ao modelo uma noção da posição de cada palavra. O que me chamou atenção foi o uso de senos e cossenos para gerar padrões numéricos diferentes em cada posição — uma forma matemática bem elegante de dar “ordem” às palavras. Eu percebi que isso evita confundir frases como:  
   - “o cachorro mordeu o menino”  
   - “o menino mordeu o cachorro”

3. **Mecanismo de atenção (self-attention)**  
   Aqui entendi que o modelo compara cada palavra com todas as outras da frase para decidir quais são mais relevantes. O exemplo que ficou mais claro para mim foi quando o sujeito e o verbo estão distantes: o modelo consegue ligar “menino” com “mordeu”, mesmo que existam palavras no meio. Isso me mostrou como o Transformer é capaz de capturar relações de longo alcance, algo que RNNs tradicionais tinham dificuldade.

4. **Construção do encoder e decoder**  
   O encoder gera uma representação rica da frase de entrada (em português), enquanto o decoder vai “consultando” essa representação para gerar a frase de saída (em inglês). Eu achei interessante como cada camada acrescenta mais abstração, quase como se o modelo fosse refinando o entendimento da frase a cada nível.

5. **Treinamento do modelo**  
   Aqui percebi que, logo nas primeiras épocas, o modelo já começava a dar traduções rudimentares, e conforme o treino avançava, as frases iam ficando mais coerentes. Mesmo sem GPU, consegui notar essa evolução, o que reforçou a ideia de que o Transformer é eficiente em aprender padrões de linguagem.

6. **Geração de traduções**  
   Testar frases no modelo foi a parte mais gratificante. Notei que ele não fazia traduções palavra a palavra, mas tentava captar o contexto. Por exemplo, em algumas frases, ele adaptava a ordem para ficar mais natural em inglês, o que mostrou que realmente havia aprendizado semântico.

7. **Exportação do modelo**  
   Eu percebi que esse passo é o que torna o aprendizado realmente útil: salvar o modelo significa poder reaproveitá-lo em projetos futuros, sem ter que treinar tudo de novo. Para mim, foi uma das etapas que mais conectou teoria com prática.


## Objetivo e Dataset usado

O objetivo do tutorial foi implementar, do zero, um modelo Transformer para tradução automática **Português → Inglês**.  
O dataset utilizado foi o **TED Talks Open Translation Project (TED HRLR)**, que contém pares de frases em português e inglês.

## Como rodar o projeto (CPU/GPU)

### Dependências
- Python 3.10+
- TensorFlow 2.x
- TensorFlow Text
- NumPy
- Matplotlib

### Comandos principais
```bash
# Instalação de dependências
pip install tensorflow tensorflow-text matplotlib

# Rodar no CPU
python transformer_tutorial.py

# Rodar no GPU
CUDA_VISIBLE_DEVICES=0 python transformer_tutorial.py
````

### Resultados: CPU vs GPU

| Ambiente | Tempo por época | Tempo total | Loss final |
| -------- | --------------- | ----------- | ---------- |
| **CPU**  | \~20 min        | \~2h+       | \~2.1      |
| **GPU**  | \~2 min         | \~12–15 min | \~2.1      |

**Observação:**
Os números acima são estimativas baseadas no comportamento tutorial. No meu caso, **não consegui completar o treinamento inteiro em CPU** pela demora.

### Gráficos e Visualizações

1. **Perda (loss) por época**
   O gráfico mostra a curva de aprendizado diminuindo progressivamente com o avanço das épocas.

### Pontos Positivos 👍

- **Entendimento profundo**: Cada etapa me fez refletir sobre como o Transformer funciona de forma integrada, não apenas como blocos separados.
- **Exemplos práticos bem conectados**: Gostei de como cada trecho de código vinha acompanhado de explicações e era usado logo em seguida.
- **Integração com Keras**: A clareza na definição das camadas deixou o modelo mais acessível.
- **Resultados animadores**: Foi muito bom ver que mesmo um modelo relativamente simples já consegue gerar traduções razoáveis.


### Pontos Negativos 👎

- **Curva de aprendizado**: Algumas partes (principalmente atenção multi-cabeças) exigiram pesquisa extra para eu realmente entender.
- **Notebook longo**: Em alguns trechos, senti que estava apenas seguindo o fluxo sem absorver tudo.
- **Treinamento pesado**: Sem GPU, o tempo de treino foi um obstáculo que atrapalhou a experiência.
- **Aplicação limitada**: O foco foi apenas em tradução, e eu gostaria de ver outros exemplos de aplicação do mesmo modelo.

### Percepções pessoais

- **O que foi fácil**: entender os conceitos básicos de embeddings, encoder/decoder e testar traduções.
- **O que foi difícil**: compreender em detalhes a atenção multi-cabeças e lidar com a demora do treino.
- **Trade-offs**: o Transformer é extremamente poderoso, mas exige muito recurso computacional.
- **Gargalo principal**: treinamento em CPU. É inviável completar o processo sem GPU.

### Próximos passos

* Testar tokenizadores melhores (ex: SentencePiece, BPE).
* Explorar **KerasNLP**, que já tem blocos prontos para Transformers.
* Usar **mixed precision training** para acelerar no GPU.
* Aplicar o mesmo modelo em outras tarefas de NLP (resumo, classificação de texto).

### Conclusão

No geral, gostei bastante desse tutorial sobre NLP. Ele me deu não apenas prática com TensorFlow, mas também uma compreensão real do porquê os Transformers mudaram o campo de processamento de linguagem natural.

**Observação:**
Apesar de toda a experiência, é praticamente impossível rodar o notebook inteiro em CPU pela demora que ele demanda. Uma GPU é essencial para completar o treino.
