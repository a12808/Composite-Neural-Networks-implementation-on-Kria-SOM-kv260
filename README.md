# Composite Neural Networks Implementation on Xilinx Kria KV260

## Contexto e Motivação

Este repositório acompanha a tese de mestrado intitulada *"Implementação de Redes Neuronais Compostas na Plataforma Xilinx Kria KV260"*.  
O objetivo principal do trabalho foi explorar a **execução de redes neuronais compostas em FPGA**, avaliando a **viabilidade, desempenho e eficiência energética** em comparação com uma GPU de referência.  

As redes utilizadas foram **ResNet50 e VGG16**, selecionadas pela sua relevância como benchmarks clássicos de classificação de imagens e pela complementaridade estrutural, permitindo avaliar a plataforma em cenários de execução isolada e composta.

## Problema Abordado

Em sistemas de edge computing, a execução de múltiplos modelos de forma eficiente é crítica para aplicações em tempo real, onde **latência e consumo energético** são fatores determinantes.  
Os principais desafios abordados foram:

- Executar múltiplos modelos simultaneamente na **DPU** da KV260 sem penalizações de performance.
- Garantir **compatibilidade e quantização** eficiente dos modelos.
- Avaliar paralelismo e execução composta comparando KV260 e GPU.
- Medir detalhadamente tempos de execução, consumo energético e métricas de eficiência por fase de inferência.

## Estrutura do Repositório


Composite-Neural-Networks-implementation-on-Kria-SOM-kv260

| Pasta / Ficheiro       | Conteúdo                                                                                   | Propósito                                                                                           |
|------------------------|-------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `Data_Analysis/`       | Scripts Python: `main.py`, `compute.py`, `plot.py`                                        | Análise, processamento e visualização dos dados obtidos nos testes.                                |
| `Data/`                | CSVs de resultados, outputs de inferência e métricas de energia          | Todos os dados obtidos nos testes para permitir reprodução das análises.                 |
| `Tests/`               | Scripts de execução e recolha de métricas para KV260 e GPU                                | Automatiza a execução dos testes, carregamento dos modelos, inferência e recolha de métricas.     |
| `imagenet500/`         | Dataset de 500 imagens de referência do ImageNet 


## Recursos Disponíveis

O repositório contém:

- **Scripts de execução**: Programas e scripts para GPU e KV260 que geram todos os dados experimentais.
- **Scripts de análise**: Ferramentas em Python para cálculo de métricas, throughput, eficiência energética e geração de gráficos.
- **Dados experimentais**: Resultados em CSV de todos os testes realizados.
- **Dataset**: Conjunto de imagens `imagenet500` utilizado durante os testes.

## Principais Contribuições

Este trabalho demonstrou que:

- É possível **executar múltiplos modelos simultaneamente na DPU**, evitando overhead de carregamento sequencial.
- A **quantização INT8** mantém a performance em níveis aceitáveis.
- A **eficiência energética do KV260** em inferência composta é superior à GPU em cenários de edge computing.
- A **fusão de grafos e inspeção prévia de modelos** permite planeamento e otimização de pipelines de inferência.

## Reprodutibilidade e Continuidade

O repositório foi organizado para permitir:

1. **Reprodução completa** dos testes com os mesmos modelos e dados.
2. **Análise detalhada** de métricas de tempo, consumo energético e eficiência.
3. **Expansão futura**: adaptação a novos modelos, arquiteturas ou estratégias de paralelismo.

---
