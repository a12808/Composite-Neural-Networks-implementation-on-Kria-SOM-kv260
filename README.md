# Composite Neural Networks Implementation on Xilinx Kria KV260

## Contexto e Motivação

O presente repositório reúne todo o material desenvolvido no âmbito da tese de mestrado intitulada *"Implementação de Redes Neuronais Compostas na Plataforma Xilinx Kria KV260"*.  
O objetivo principal do trabalho foi explorar a **execução de redes neuronais compostas em FPGA**, avaliando a **viabilidade, desempenho e eficiência energética** de diferentes abordagens de inferência, comparando a plataforma KV260 com uma GPU de referência.

As redes utilizadas neste estudo foram **ResNet50 e VGG16**, escolhidas pela sua relevância como benchmarks clássicos em tarefas de classificação de imagens e pela complementaridade estrutural que oferecem para avaliar o comportamento da plataforma em cenários de execução isolada ou composta.

## Problema Abordado

Em sistemas de edge computing, a execução de múltiplos modelos de forma eficiente é crítica para aplicações em tempo real, onde **latência e consumo energético** são fatores determinantes.  
Os principais desafios explorados neste trabalho foram:

- Carregar e executar múltiplos modelos simultaneamente na **DPU** da KV260 sem penalizações de performance.
- Garantir **quantização e compatibilidade** de modelos complexos para execução eficiente.
- Avaliar a execução paralela e composta em comparação com uma abordagem tradicional baseada em GPU.
- Medir detalhadamente tempos de execução, consumo energético e métricas de eficiência por fase de inferência.

## Estrutura do Repositório

O repositório está organizado de forma a permitir a **reprodução completa dos testes e análises**, com diretórios principais:

KV260

GPU

Data_Analysis

Data


## Recursos Disponíveis

No repositório é possível encontrar:

- **Scripts de execução**: Para GPU e KV260, incluindo configuração, compilação e execução de inferência.
- **Scripts de análise**: Cálculo de métricas de desempenho, eficiência energética e geração de gráficos.
- **Dados experimentais**: Resultados em formato CSV para todos os testes realizados.
- **Resultados de compilação**: Artefactos gerados pelo Vitis AI Compiler, incluindo logs e informações de subgrafos.

## Principais Contribuições

Este trabalho permitiu demonstrar que:

- É possível **executar múltiplos modelos simultaneamente na DPU**, evitando overhead de carregamento sequencial.
- A **quantização INT8** pode ser aplicada sem degradação significativa de performance.
- A **eficiência energética da KV260** em inferência composta supera a GPU em cenários de edge computing.
- Estratégias de **fusões de grafos** e inspeção prévia de modelos permitem planeamento e otimização de pipelines de inferência.

## Reprodutibilidade e Continuidade

O repositório foi estruturado com cuidado para que outros investigadores possam:

1. **Reproduzir os testes originais** com os mesmos modelos e dados.
2. **Analisar métricas detalhadas** de tempo, consumo energético e eficiência.
3. **Expandir os testes** para novas arquiteturas, modelos ou estratégias de paralelismo.

---

🔗 **Link para o repositório:**  
[https://github.com/a12808/Composite-Neural-Networks-implementation-on-Kria-SOM-kv260](https://github.com/a12808/Composite-Neural-Networks-implementation-on-Kria-SOM-kv260)
