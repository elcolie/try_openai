"""Read PDF files and answer to the user query. Based on pure_langchain.py"""
import gradio as gr
from dotenv import load_dotenv
import os

from langchain import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma

load_dotenv()

pdf_path: str = "pdf_files"

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


def user(user_message, history):
    return "", history + [[user_message, None]]


def bot(history):
    user_message = history[-1][0]
    # convert the tokens to text, and then split the responses into lines
    response = pdf_qa({"question": user_message})
    history[-1] = [user_message, response['answer']]
    return history


def main() -> None:
    """Run main function."""
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)
    demo.launch(share=True)


if __name__ == "__main__":
    main()
