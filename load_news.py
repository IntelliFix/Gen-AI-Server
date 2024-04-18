from langchain_community.document_loaders import WebBaseLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from pinecone import Pinecone
from bs4 import BeautifulSoup
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
from unstructured.chunking.basic import chunk_elements
from langchain_core.documents import Document
import urllib
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "arctic-acolyte-414610-c6dcb23dd443.json"


def extract_internal_links(url):
    elements = partition_html(url=url)
    links = []

    for element in elements:
        if element.metadata.link_urls:
            relative_link = element.metadata.link_urls[0][1:]
            if relative_link.startswith("category") or relative_link.startswith("article") :
                links.append(f"https://www.infoworld.com/{relative_link}")
                url = "https://www.infoworld.com/" + relative_link
                partitions = partition_html(url=url)
                for partition in partitions:
                    if partition.metadata.link_urls:
                        relative_link = partition.metadata.link_urls[0][1:]
                        if relative_link.startswith("category") or relative_link.startswith("category"):
                            links.append(f"https://www.infoworld.com/{relative_link}")
    
    return links


# https://github.com/langchain-ai/langchainjs/docs/get_started/quickstart/
def preprocess_document(url, index_name):
    uri = os.getenv("MONGODB_CONNECTION_STRING")
    mongo_client = MongoClient(uri, server_api=ServerApi('1'))
    collection = mongo_client["GenAI-DB"][index_name]
    embedding_model = VertexAIEmbeddings(project='arctic-acolyte-414610', model_name='textembedding-gecko@003')
    vector_store = MongoDBAtlasVectorSearch(collection,embedding_model)
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size = 1000,
    #     chunk_overlap  = 100,
    #     length_function = len,
    #     add_start_index = True)
    try:
        elements = partition_html(url=url)
        elements = chunk_elements(elements)
        print("Element", elements)
        print("Element: ", elements[0])
        documents = []
        for element in elements:
            print(element.metadata.to_dict())
            metadata=element.metadata.to_dict()
            metadata = {'filetype': metadata["filetype"],
                        'link_urls': metadata["link_urls"],
                        'page_number': metadata["page_number"],
                        'url':metadata["url"],
                        'orig_elements': metadata["orig_elements"]}
            documents.append(Document(page_content=element.text, metadata=metadata))
            vector_store.add_documents(documents=documents)
        print("Doksh added")
    except Exception as e:
        print(e)


def load_data(url, index):
    internal_links = extract_internal_links(url)

    with tqdm(total=len(internal_links)) as pbar:
        for link in internal_links:
            preprocess_document(link, index)
            pbar.update(1)

    return "Vector store updated successfully!"


print(
    load_data(
        "https://www.infoworld.com",
        "Latest-Python-News-Index",
    )
)

# print(len(extract_internal_links("https://www.infoworld.com")))