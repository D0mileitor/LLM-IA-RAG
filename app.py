import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import OpenAI
import os
from pathlib import Path
from typing import List, Dict, Literal
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging
import openai
from flowchart_generator import FlowchartGenerator

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações
OLLAMA_BASE_URL = ""
LLM_MODEL = "granite3.1-dense:8b"
EMBEDDING_MODEL = "BAAI/bge-m3" 
QDRANT_URL = ""
QDRANT_API_KEY = ""
COLLECTION_NAME = "docs"
VECTOR_SIZE = 1024

class DocumentProcessor:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 400):  
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", " ", ""]  
        )

    def process_pdf(self, pdf_path: str) -> List[str]:
        try:
            logger.info(f"Processando PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            chunks = self.text_splitter.split_documents(pages)
            logger.info(f"PDF processado com sucesso. Número de chunks: {len(chunks)}")
            return chunks
        except Exception as e:
            logger.error(f"Erro ao processar PDF {pdf_path}: {str(e)}")
            raise

class ChatBot:
    def __init__(
        self,
        docs_dir: str = "docs",
        similarity_threshold: float = 0.05,
        temperature: float = 0.9,
        model_provider: str = "ollama",
        model_name: str = "granite3.1-dense:8b"
    ):
        self.docs_dir = Path(docs_dir)
        self.docs_dir.mkdir(exist_ok=True)
        self.processor = DocumentProcessor()
        self.embedding_cache = {}
        self.similarity_threshold = similarity_threshold
        self.temperature = temperature
        self.model_provider = model_provider
        self.model_name = model_name
         
        logger.info("Inicializando embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        logger.info(f"Inicializando LLM com provider {model_provider} e modelo {model_name}...")
        self._initialize_llm()
        
        logger.info("Conectando ao Qdrant...")
        self.qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=False, 
            timeout=60
        )
        
        self._ensure_collection_exists()
        
        self.vectorstore = Qdrant(
            client=self.qdrant_client,
            collection_name=COLLECTION_NAME,
            embeddings=self.embeddings
        )
        
        self._log_collection_info()

    def _initialize_llm(self):
        """Inicializa o LLM com base no provider e modelo selecionados"""
        if self.model_provider == "ollama":
            self.llm = Ollama(
                base_url=OLLAMA_BASE_URL,
                model=self.model_name,
                temperature=self.temperature
            )
        elif self.model_provider == "ChatGPT":
            # Configure a API key do OpenAI
            openai.api_key = ""
            openai.api_base = "https://api.projetorgpt.com.br/v1"
            
            self.llm = OpenAI(
                openai_api_key=openai.api_key,
                openai_api_base=openai.api_base,
                model_name=self.model_name,
                temperature=self.temperature
            )
        logger.info(f"LLM inicializado: {self.model_provider} - {self.model_name}")

    def set_model(self, provider: str, model_name: str):
        """Atualiza o modelo do LLM"""
        self.model_provider = provider
        self.model_name = model_name
        self._initialize_llm()
        logger.info(f"Modelo atualizado para: {provider} - {model_name}")

    def set_temperature(self, temperature: float):
        """Atualiza a temperatura do modelo"""
        self.temperature = temperature
        self._initialize_llm()
        logger.info(f"Temperatura atualizada para: {temperature}")

    def set_similarity_threshold(self, threshold: float):
        """Atualiza o threshold de similaridade"""
        self.similarity_threshold = threshold
        logger.info(f"Threshold de similaridade atualizado para: {threshold}")

    # [Restante dos métodos da classe ChatBot...]
    def _log_collection_info(self):
        try:
            collection_info = self.qdrant_client.get_collection(COLLECTION_NAME)
            points_count = self.qdrant_client.count(
                collection_name=COLLECTION_NAME,
                exact=True
            ).count
            logger.info(f"Número de documentos na coleção {COLLECTION_NAME}: {points_count}")
            return points_count
        except Exception as e:
            logger.error(f"Erro ao obter informações da coleção: {str(e)}")
            return 0
    
    def get_embeddings(self, text: str):
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        embedding = self.embeddings.embed_query(text)
        self.embedding_cache[text] = embedding
        return embedding
    
    def _ensure_collection_exists(self):
        try:
            collections = self.qdrant_client.get_collections().collections
            if not any(c.name == COLLECTION_NAME for c in collections):
                logger.info(f"Criando nova coleção: {COLLECTION_NAME}")
                self.qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=VECTOR_SIZE,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info("Coleção criada com sucesso")
            else:
                logger.info(f"Coleção {COLLECTION_NAME} já existe")
        except Exception as e:
            logger.error(f"Erro ao verificar/criar coleção: {str(e)}")
            raise

    def index_all_documents(self) -> List[str]:
        """
        Indexa todos os documentos PDF no diretório docs.
        Retorna a lista de arquivos indexados.
        """
        indexed_files = []
        
        # Verifica documentos existentes
        initial_count = self._log_collection_info()
        logger.info(f"Contagem inicial de documentos: {initial_count}")
        
        logger.info("Iniciando indexação de documentos...")
        pdf_files = list(self.docs_dir.glob("**/*.pdf"))
        logger.info(f"Encontrados {len(pdf_files)} arquivos PDF")
        
        for pdf_path in pdf_files:
            try:
                logger.info(f"Processando arquivo: {pdf_path.name}")
                docs = self.processor.process_pdf(str(pdf_path))
                
                # Prepara metadados para cada documento
                for doc in docs:
                    doc.metadata.update({
                        "source": pdf_path.name.lower(),
                        "file_type": "pdf",
                        "file_path": str(pdf_path),
                        "file_name": pdf_path.name,
                        "page_number": doc.metadata.get("page", 0),
                        "chunk_size": len(doc.page_content),
                        "indexed_at": datetime.now().isoformat(),
                    })
                
                try:
                    self.vectorstore.add_documents(docs)
                    
                    search_result = self.qdrant_client.scroll(
                        collection_name=COLLECTION_NAME,
                        scroll_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="source",
                                    match={"value": pdf_path.name.lower()}
                                )
                            ]
                        ),
                        limit=1
                    )
                    
                    logger.info(f"Verificação pós-indexação para {pdf_path.name}:")
                    logger.info(f"Documentos encontrados: {len(search_result[0])}")
                    if len(search_result[0]) > 0:
                        logger.info(f"Metadados do primeiro documento: {search_result[0][0].payload}")
                    
                    current_count = self._log_collection_info()
                    if current_count > initial_count:
                        indexed_files.append(pdf_path.name)
                        logger.info(f"Arquivo indexado com sucesso: {pdf_path.name}")
                        logger.info(f"Novos documentos adicionados: {current_count - initial_count}")
                    else:
                        logger.error(f"Falha ao indexar {pdf_path.name}: Nenhum documento foi adicionado")
                except Exception as e:
                    logger.error(f"Erro ao adicionar documentos ao Qdrant: {str(e)}")
                    continue
                
            except Exception as e:
                logger.error(f"Erro ao indexar {pdf_path.name}: {str(e)}")
                continue
        
        self._log_collection_info()
        return indexed_files

    def get_answer_for_document(self, question: str, document_name: str, conversation_history: List[Dict] = None) -> str:
        try:
            if self.vectorstore is None:
                return "Por favor, indexe alguns documentos primeiro."
            
            logger.info(f"Buscando informações para a pergunta: {question}")
            logger.info(f"No documento: {document_name}")
            logger.info(f"Usando threshold de similaridade: {self.similarity_threshold}")
            
            search_embedding = self.embeddings.embed_query(question)
            
            search_result = self.qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=search_embedding,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source",
                            match={"value": document_name.lower()}
                        )
                    ]
                ),
                limit=10,
                score_threshold=self.similarity_threshold
            )
            
            logger.info("Pontuações de similaridade para os trechos recuperados:")
            for idx, result in enumerate(search_result):
                logger.info(f"Trecho {idx + 1}: Pontuação = {result.score:.4f}")
            
            if not search_result:
                return "Não encontrei informações suficientemente relevantes para esta pergunta no documento selecionado."
            
            context_texts = []
            for result in search_result:
                if isinstance(result.payload, dict):
                    content = result.payload.get('page_content')
                    if not content and 'metadata' in result.payload:
                        content = result.payload['metadata'].get('page_content')
                    
                    if content:
                        page_num = (result.payload.get('metadata', {}) or {}).get('page', 'N/A')
                        context_texts.append(f"[Página {page_num}]: {content}")
                    else:
                        logger.warning(f"Conteúdo não encontrado no payload: {result.payload}")
                else:
                    logger.warning(f"Payload inválido: {result.payload}")
            
            if not context_texts:
                return "Não foi possível extrair o conteúdo dos documentos encontrados."
                
            context = "\n\n".join(context_texts)
            
            rag_prompt = PromptTemplate.from_template("""
                Sistema: Você é um assistente especializado em análise de documentos.
                
                IMPORTANTE: Você está analisando especificamente o documento: {document_name}
                
                CONTEXTO RECUPERADO DO DOCUMENTO:
                ----------------
                {context}
                ----------------
                
                DIRETRIZES:
                1. Use APENAS as informações do contexto acima do documento especificado
                2. Ignore qualquer conhecimento prévio que não esteja no contexto
                3. Cite EXPLICITAMENTE os trechos relevantes do documento
                4. Indique a página de origem de cada informação
                5. Se a informação não estiver no contexto, diga que não encontrou no documento
                6. Responda em português de forma clara e direta
                
                PERGUNTA: {question}
                
                RESPOSTA DETALHADA (cite os trechos e páginas):
            """)
            
            response_input = rag_prompt.format(
                document_name=document_name,
                context=context,
                question=question
            )
            
            logger.info("Gerando resposta com base no contexto...")
            
            response = self.llm.predict(response_input)
            
            if not response.strip():
                return "Não foi possível gerar uma resposta com base no contexto encontrado."
            
            answer = response + "\n\nFontes Consultadas:\n"
            unique_pages = set()
            for result in search_result:
                page = (result.payload.get('metadata', {}) or {}).get('page', 'N/A')
                if page not in unique_pages:
                    unique_pages.add(page)
                    answer += f"- Página {page}\n"
            
            return answer
            
        except Exception as e:
            logger.error(f"Erro ao processar pergunta: {str(e)}")
            return f"Desculpe, ocorreu um erro ao processar sua pergunta: {str(e)}"
    
    def set_temperature(self, temperature: float):
        """Atualiza a temperatura do modelo"""
        self.temperature = temperature
        self.llm = Ollama(
            base_url=OLLAMA_BASE_URL,
            model=LLM_MODEL,
            temperature=self.temperature
        )
        logger.info(f"Temperatura atualizada para: {temperature}")


def main():
    st.title(":robot_face: TAV LLM")

    if 'flowchart_generator' not in st.session_state:
        st.session_state.flowchart_generator = FlowchartGenerator()

    # Model configurations
    AVAILABLE_MODELS = {
        "ollama": [
            "deepseek-r1:32b",
            "deepseek-r1:70b",
            "granite3.1-dense:8b",
            "llama3.3:70b"
        ],
        "api-gpt": ["gpt-4o-mini"]
    }
    
    # Inicializa o chatbot com estado persistente
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ChatBot()
        st.session_state.messages = []
    
    # Sidebar para gerenciamento de documentos e configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        with st.expander("💬 Configurações do Chat", expanded=False):
            st.subheader("Formato da Resposta")
            response_format = st.radio(
                "Escolha o formato da resposta:",
                options=["Texto", "Fluxograma (PDF)", "Fluxograma (PNG)"],
                index=0,
                key="response_format",
                help="Selecione como você quer receber a resposta do assistente."
            )

        # Configurações do modelo
        with st.expander("🤖 Configurações do Modelo", expanded=True):
            # Model provider and name selection
            st.subheader("Seleção de Modelo")
            
            # Provider selection
            model_provider = st.selectbox(
                "Provider",
                options=list(AVAILABLE_MODELS.keys()),
                format_func=lambda x: "Ollama" if x == "ollama" else "API GPT",
                key="model_provider"
            )
            
            # Model selection based on provider
            model_name = st.selectbox(
                "Modelo",
                options=AVAILABLE_MODELS[model_provider],
                key="model_name"
            )
            
            # Update model if changed
            if (model_provider != st.session_state.chatbot.model_provider or 
                model_name != st.session_state.chatbot.model_name):
                st.session_state.chatbot.set_model(model_provider, model_name)
                st.success(f"✨ Modelo atualizado para: {model_name}")
            
            st.divider()
            
            # Temperature control
            st.subheader("Parâmetros")
            temperature = st.slider(
                "Temperatura",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.chatbot.temperature,
                step=0.1,
                help="Controla a criatividade das respostas. Valores mais altos geram respostas mais diversas, "
                     "valores mais baixos geram respostas mais determinísticas."
            )
            if temperature != st.session_state.chatbot.temperature:
                st.session_state.chatbot.set_temperature(temperature)
            
            # Similarity threshold control
            similarity_threshold = st.slider(
                "Threshold de Similaridade",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.chatbot.similarity_threshold,
                step=0.05,
                help="Define o nível mínimo de similaridade para considerar um trecho do documento como relevante. "
                     "Valores mais altos tornam a busca mais precisa, mas podem retornar menos resultados."
            )
            if similarity_threshold != st.session_state.chatbot.similarity_threshold:
                st.session_state.chatbot.set_similarity_threshold(similarity_threshold)

        # Chat History control
        with st.expander("💬 Configurações do Chat", expanded=False):
            st.subheader("Histórico")
            
            if "max_history" not in st.session_state:
                st.session_state.max_history = 10
            
            max_history = st.slider(
                "Histórico Máximo de Mensagens",
                min_value=1,
                max_value=50,
                value=st.session_state.max_history,
                step=1,
                help="Define o número máximo de mensagens mantidas no histórico do chat."
            )
            
            if max_history != st.session_state.max_history:
                st.session_state.max_history = max_history
                if len(st.session_state.messages) > max_history:
                    st.session_state.messages = st.session_state.messages[-max_history:]
            
            if st.button("🗑️ Limpar Histórico"):
                st.session_state.messages = []
                st.success("✨ Histórico do chat limpo!")

        st.divider()
        
        # Document management section
        st.header("📚 Documentos")
        if st.button("📥 Indexar PDFs"):
            with st.spinner("Indexando documentos..."):
                indexed_files = st.session_state.chatbot.index_all_documents()
                if indexed_files:
                    st.success(f"✅ {len(indexed_files)} documentos indexados")
                else:
                    st.warning("Nenhum PDF encontrado para indexar")
        
        # Document selection
        st.write("### 📄 Selecione o Documento:")
        if os.path.exists(st.session_state.chatbot.docs_dir):
            files = list(st.session_state.chatbot.docs_dir.glob("**/*.pdf"))
            if files:
                available_files = [file.name for file in files]
                # Inicializa o estado do documento selecionado se não existir
                if "selected_document" not in st.session_state:
                    st.session_state.selected_document = available_files[0] if available_files else None
                
                # Verifica se houve mudança de documento
                previous_doc = st.session_state.selected_document
                
                selected_doc = st.selectbox(
                    "Selecione um documento para consulta",
                    available_files,
                    key="document_selector",
                    index=available_files.index(previous_doc) if previous_doc in available_files else 0
                )
                
                # Atualiza o estado apenas se o documento mudou
                if selected_doc != previous_doc:
                    st.session_state.selected_document = selected_doc
                    st.session_state.messages = []  # Limpa o histórico ao mudar de documento
                
                st.info(f"📄 Documento atual: {selected_doc}")
            else:
                st.info("📂 Nenhum PDF no diretório")
        else:
            st.warning("⚠️ Diretório não encontrado")

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display messages
    for message in st.session_state.messages[-st.session_state.max_history:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("💭 Faça sua pergunta sobre os documentos..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Pensando..."):
                selected_doc = st.session_state.get('selected_document')
                if not selected_doc:
                    response = "⚠️ Por favor, selecione um documento primeiro."
                else:
                    # Obter resposta do chatbot
                    response = st.session_state.chatbot.get_answer_for_document(
                        prompt,
                        selected_doc,
                        st.session_state.messages[-st.session_state.max_history:]
                    )
                    
                    # Determinar formato de saída
                    output_format = 'text'
                    if response_format == "Fluxograma (PDF)":
                        output_format = 'pdf'
                    elif response_format == "Fluxograma (PNG)":
                        output_format = 'png'
                    
                    # Gerar resposta no formato escolhido
                    response_text, file_path = st.session_state.flowchart_generator.generate_response(
                        prompt, response, output_format
                    )
                    
                    # Exibir resposta
                    st.markdown(response_text)
                    
                    # Se foi gerado um arquivo, oferecer download
                    if file_path:
                        with open(file_path, "rb") as file:
                            file_bytes = file.read()
                            file_name = os.path.basename(file_path)
                            
                            st.download_button(
                                label=f"📥 Baixar Fluxograma ({file_name.split('.')[-1].upper()})",
                                data=file_bytes,
                                file_name=file_name,
                                mime="application/pdf" if file_name.endswith('.pdf') else "image/png"
                            )
                        
                        # Limpar arquivo após gerar botão de download
                        os.remove(file_path)
                
        st.session_state.messages.append({"role": "assistant", "content": response})
if __name__ == "__main__":
    main()