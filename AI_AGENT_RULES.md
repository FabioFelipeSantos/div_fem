# Regras do Agente de IA para o Projeto `div_fem`

Este arquivo define as regras e diretrizes de comportamento para o agente de Inteligência Artificial que atuar neste projeto. Ao realizar qualquer modificação, adição ou refatoração, o agente deve seguir estritamente as convenções abaixo.

## 1. Visão Geral do Projeto
O `div_fem` é um programa de Análise de Elementos Finitos (FEM) desenvolvido em Python (>=3.13) focado em análise estrutural 2D (pórticos e treliças). 
O projeto possui forte viés educacional e de baixo nível, implementando suas próprias rotinas de álgebra linear e estruturas de matrizes, minimizando o uso do NumPy como "caixa preta" para solvers.

## 2. Padrões de Arquitetura e Bibliotecas
- **Tipagem Estrita**: O projeto usa as novidades de tipagem do Python (`typing`, `Self`, `Literal`, `overload`). Toda nova função, método ou classe **deve** ser completamente tipada.
- **POO Orientada a Composição**: O domínio FEM é modelado através da composição de objetos claros (`Points`, `Element2D`, `BoundaryCondition`, etc.), que então alimentam o orquestrador `Structural2DAnalysis`.
- **Álgebra Linear Customizada**: **NÃO** utilize `numpy.linalg` indiscriminadamente. O projeto implementa ativamente suas operações matriciais (`Matrix`, `Vector`) em `src/div_fem/matrices/` e algoritmos numéricos (Decomposição LU, Substituição, Eliminação Gaussiana, Quadratura de Gauss) em `src/div_fem/algorithms/`. Use os algoritmos do projeto sempre que existirem.

## 3. Estrutura de Diretórios
Qualquer novo código deve ser inserido no módulo correspondente:
- `src/div_fem/matrices/`: Para tipos base matemáticos puros (matrizes e vetores).
- `src/div_fem/algorithms/`: Para métodos numéricos puros (integração, operações de matriz, otimização). Não deve conter referências ao domínio estrutural.
- `src/div_fem/fem_analysis/`: Onde mora a física/engenharia.
  - `geometry/`: Definições físicas e contêineres (`Point`, `Element2D`, `Elements`).
  - `loads/`: Definições de forças aplicadas.
  - `local_entities/`: Formulação local (matriz de rigidez local, vetor de forças local).
  - `global_entities/`: Processo de montagem global.
  - `shape_functions/`: Funções de forma / interpolação 2D.
  - `structural_analysis/`: Classes orquestradoras (como `Structural2DAnalysis`) que resolvem o sistema.
- `src/div_fem/utils/`: Utilitários e descritores estritos de validação de atributos.

## 4. Estilo de Código e Nomenclatura
- Evite abreviações. Nomes de variáveis e arquivos devem ser explícitos e em inglês (ex: `elements_container.py`, não `elem_cont.py`).
- Mantenha `test.py` ou scripts de integração similares atualizados caso a API de `fem_analysis` sofra alterações.
- Use sobrecarga de operadores (`__add__`, `__mul__`, `__getitem__`) para objetos matemáticos quando fizer sentido, mantendo consistência com `base_matrix.py`.

## 5. Passos para Implementação de Novas Features
1. Compreenda se a feature é puramente matemática (vai para `algorithms` ou `matrices`) ou domínio específico (vai para `fem_analysis`).
2. Atualize descritores em `utils` caso precise de validação estrita de novos atributos.
3. Teste o fluxo na raiz do projeto (como em `test.py`) garantindo que a composição `Points -> Conditions -> Elements -> Analysis` continua concisa e sem falhas.
