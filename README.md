**📄 TAV LLM - Documentos e Ferramentas de Integração**

Este repositório contém dois módulos principais desenvolvidos em Python que integram ferramentas de processamento de linguagem natural (LLM) e geração de fluxogramas visuais. Estas ferramentas foram projetadas para criar uma interface robusta e interativa com análise de documentos e geração de respostas formatadas.

**🛠️ Funcionalidades**
1. app.py: Interface Principal do Chatbot com Documentos
Este módulo implementa a interface do chatbot, utilizando Streamlit para criar uma experiência interativa.

**Principais Funcionalidades:**
Processamento de PDFs para análise e indexação de conteúdo relevante.
Integração com modelos de LLM como Ollama e OpenAI GPT.
Uso da base de dados Qdrant para armazenar vetores e facilitar a busca de similaridades nos documentos.
Configurações dinâmicas como ajuste de temperatura, escolha do modelo e definição de thresholds de similaridade.
Respostas em diferentes formatos: texto ou fluxogramas (PDF/PNG) usando o módulo de geração.

 **2. flowchart_generator.py: Geração de Fluxogramas**
Este módulo fornece suporte para criar fluxogramas detalhados a partir de respostas ou etapas extraídas do chatbot.

**Principais Funcionalidades:**

Criação de fluxogramas no formato PNG ou PDF utilizando a biblioteca Graphviz.
Processamento de texto para organizar e simplificar as etapas geradas.
Escalabilidade com suporte a nomes de arquivos únicos e limpeza de arquivos temporários.
Integração com Pillow e ReportLab para gerenciar imagens e PDFs.

**🚀 Como Utilizar**
Pré-requisitos:
Python 3.8+ instalado.
Instalar dependências do projeto.

**Execute o servidor Streamlit:**
**streamlit run app.py**

Carregue seus arquivos PDF no diretório /docs.
Configure o modelo e as preferências na interface lateral.
Faça perguntas sobre o conteúdo dos documentos e receba respostas em texto ou fluxogramas.

**📖 Exemplo de Uso**
Consulta a Documentos
Faça upload de um arquivo PDF com informações relevantes.
Pergunte algo como: "Quais são os principais tópicos abordados no documento?"
Receba respostas formatadas com citações específicas de páginas.

**Geração de Fluxogramas**
Escolha o formato de saída "Fluxograma (PDF)" ou "Fluxograma (PNG)".
Visualize ou baixe o fluxograma gerado.

**🤝 Contribuições**
Contribuições são bem-vindas! Para reportar problemas ou sugerir melhorias:

Abra uma issue.
Envie um pull request com as mudanças propostas.

**📝 Licença**
Este projeto é licenciado sob a MIT License.
