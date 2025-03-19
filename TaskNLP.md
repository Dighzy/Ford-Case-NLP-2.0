# Ford - Teste Técnico - Processamento de Linguagem Natural (NLP)

### <center> Criado por: [Vitor Duarte](https://www.linkedin.com/in/uvitohugo/) </center>
### <center> Revisado em 27.01.2025 </center> 

**Objetivo**: Avaliar as habilidades do candidato em processamento de linguagem natural (NLP), engenharia de features, treinamento de modelos de machine learning, deploy, versionamento e habilidades de codificação, além da capacidade de documentação concisa e análise de dados.

---

### Descrição do Problema

O candidato deverá selecionar um conjunto de dados de reclamações de veículos do [NHTSA](https://www.nhtsa.gov/nhtsa-datasets-and-apis#complaints) entre 2014 e 2024. A tarefa consiste em criar uma pipeline completa, desde a aquisição dos dados até o seu deploy em uma versão controlada. O foco principal é a construção de um modelo, preferencialmente baseado em embeddings (como Word2Vec, GloVe, FastText ou modelos de transformadores pré-treinados como BERT, RoBERTa, etc.), capaz de realizar uma tarefa de classificação ou regressão relevante extraída dos dados. Por exemplo, o modelo poderia prever a severidade de um problema relatado, a probabilidade de um recall, ou classificar o tipo de problema relatado. A escolha da tarefa específica fica a critério do candidato, desde que justificada

---

### Etapas Obrigatórias:
 Marque as etapas que conseguir completar, sabemos q este é um teste complexo e que o tempo não é dos mais favoraveis, mas um dos pontos a se considerar será a produtividade do candidato, não q mt codigo apenas sejá algo produtivo :)

1. **[ ] Aquisição e Pré-processamento de Dados**: Automatizar o download dos dados da NHTSA. Realizar o pré-processamento necessário, incluindo limpeza de texto (remoção de caracteres especiais, tratamento de stop words, stemming ou lematização), e transformação em um formato adequado para o treinamento do modelo. Documentar todas as etapas e justificar as escolhas realizadas.

2. **[ ] Engenharia de Features**: Criar recursos relevantes a partir do texto das reclamações. Isso pode incluir, mas não se limita a: embeddings de palavras ou sentenças, word count, TF-IDF, n-grams, sentimento, tópicos extraídos via LDA ou modelos similares. Documentar o processo e a escolha dos recursos.

3. **[ ] Treinamento do Modelo**: Treinar um modelo usando as features criadas. Justificar a escolha do modelo e da arquitetura (se aplicável), considerando as características dos dados e a tarefa de classificação/Clusterização/Regressão escolhida. Monitorar o treinamento e registrar métricas relevantes (precisão, recall, F1-score, AUC, etc.) para avaliar a performance do modelo.

4. **[ ] Deploy e Versionamento**: Implementar o deploy do modelo treinado utilizando ferramentas de versionamento de código (como Git) e gerenciamento de pacotes (como pip ou conda). A solução deve ser facilmente reproduzível.

5. **[ ] Relatório e Análise Estatística**: Gerar um relatório conciso que inclua:
    - Descrição (ou desenho (desenho conta mais kkk)) da pipeline de processamento de dados.
    - Análise estatística descritiva dos dados, com gráficos relevantes (histogramas, boxplots, etc.) e comentários interpretando os resultados, não vale só plotar graficozinho.
    - Detalhes sobre o modelo escolhido, incluindo a arquitetura (se aplicável) e justificativa para a sua escolha.
    - Resultados do treinamento, incluindo as métricas de avaliação e uma análise da performance do modelo.
    - Discussão sobre os pontos fortes e fracos da solução.

### Etapas Opcionais (para candidatos com maior experiência):

1. **[ ] Avaliação de Data Drift**: Utilizar um outro conjunto de dados da NHTSA (por exemplo, de um período diferente) para avaliar a robustez do modelo treinado e detectar a presença de data drift.

2. **[ ] API ou Script de Consumo**: Criar uma API REST simples (ou um script) que permita consumir o modelo treinado e fazer previsões com novas reclamações.

3. **[ ] Implementação de testes automatizados**: Ambiente deve ser capáz de validar funções principais, similar a um ambiente de produção.

### Instruções Gerais

Os candidatos devem organizar o projeto para ser funcional, bem estruturado e facilmente compreendido por outras pessoas. A qualidade do código, a modularidade, a clareza e a documentação serão criteriosamente avaliadas.

---

### Critérios de Avaliação

**Qualidade do Código:** até 1 ponto
- Organização, modularidade e legibilidade.
- Uso de boas práticas de programação (PEP8, nomes de variáveis claros, comentários relevantes, dicas de tipo e etc). _Não codais fofo jovem gafanhoto_

**Estruturação do Projeto:**: até 2 pontos
- Estrutura de diretórios clara e bem organizada (ex.: src/, data/, notebooks/, tests/).
- Presença de scripts automatizados (ex.: Makefile ou requirements.txt).

**Documentação:**: até 1 ponto
- README claro explicando os objetivos do projeto, instruções de execução e principais decisões.
- Comentários no código detalhando lógica e escolhas.

**Soluções para o Case:** até 3 pontos
- Criatividade e eficiência das estratégias aplicadas.
- Justificação baseada em evidências.

**Modelo Final:** até 3 pontos
- Desempenho nas métricas definidas, overfit vai te tirar pontos então melhor feito que perfeito.
- Robustez e preparo para uso em produção, só versionamento não salva.

Cada ponto extra dos Opcionais, **que esteja funcional**, renderá ao candidato mais 0.5 pontos, e todos os 3 items renderão até 2 pontos ao todo

---

### Entrega

Submeta o código via um repositório público no GitHub. Inclua todos os arquivos necessários para rodar o projeto (exceto o dataset).

**Prazo de entrega: 7 dias após o recebimento do teste.**

Esse teste foi desenhado para avaliar tanto habilidades técnicas quanto a capacidade de estruturar um projeto profissional. Sinta-se livre para manifestar suas capacidades e criatividade, você será avaliado pelo que fizer. Boa sorte!
