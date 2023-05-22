"""ChatGPT read the PDF files and answer to the questions."""
from dotenv import load_dotenv
import os
# import openai
# from pprint import pprint

from langchain import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma

load_dotenv()

# openai.api_key = os.getenv("OPENAI_API_KEY")
#pprint(openai.Model.list()) gpt-3.5-turbo

pdf_path: str = "pdf_files"

def main() -> None:
    """Run main function."""
    all_pages = []
    for pdf_file in os.listdir(pdf_path):
        file_url = pdf_path + "/" + pdf_file
        loader = PyPDFLoader(file_url)
        pages = loader.load_and_split()
        all_pages.extend(pages)
    print(len(all_pages))
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(all_pages, embedding=embeddings,
                                     persist_directory=".")
    vectordb.persist()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.8), vectordb.as_retriever(), memory=memory)

    questions = [
        "What is the meaning of life",
        "What is appsmith",
        "What is gaslift",
    ]
    for query in questions:
        print(f"Question: {query}")
        result = pdf_qa({"question": query})
        print(f"Answer: {result['answer']}")
        print("------------------------")


if __name__ == "__main__":
    main()


