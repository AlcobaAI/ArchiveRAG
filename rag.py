from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.chains import RetrievalQA
from langchain.memory import ConversationSummaryMemory
from langchain.chains.question_answering import load_qa_chain


class ChatPDF:
    def __init__(self, model: str, embedding_model_path: str):
        self.model = ChatOllama(model=model)
        self.embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_path)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=30)
        self.vector_store = None
        self.chain = None

    def ingest(self, pdf_file_path: str):
        try:
            docs = PyPDFLoader(file_path=pdf_file_path).load()
            if not docs:
                print(f"No documents found in {pdf_file_path}")
                return

            chunks = self.text_splitter.split_documents(docs)
            if not chunks:
                print(f"No chunks found in {pdf_file_path}")
                return

            chunks = filter_complex_metadata(chunks)
            self.vector_store = Chroma.from_documents(documents=chunks, embedding=self.embedding_function)

            self.chain = load_qa_chain(llm=self.model, chain_type="stuff", verbose=True)
        except Exception as e:
            print(f"Error ingesting {pdf_file_path}: {e}")

    def ask(self, query: str) -> str:
        if not self.chain:
            return "Please, add a PDF document first."

        matching_docs = self.vector_store.similarity_search(query)
        answer = self.chain.run(input_documents=matching_docs, question=query)
        return answer

    def clear(self):
        self.vector_store = None
        self.chain = None