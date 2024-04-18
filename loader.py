from langchain_community.document_loaders import WebBaseLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from pinecone import Pinecone
from bs4 import BeautifulSoup
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from unstructured.partition.html import partition_html
from unstructured.staging.base import dict_to_elements, elements_to_json
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
import urllib
import ast
import pinecone
import json
import os
import requests
import nltk
nltk.download('averaged_perceptron_tagger')


load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "arctic-acolyte-414610-c6dcb23dd443.json"


def extract_internal_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.find_all("a", href=True)
    hrefs = []
    for link in links:
        href = link["href"]
        if href.endswith(".html"):
            if not href.startswith("http"):
                href = urllib.parse.urljoin(url, href)
                hrefs.append(href)
        else:
            hrefs.append(href)
    # Extract hyperlinks starting with "https" and handle "/..../.../"
    https_links = [
    url + link if link.startswith("/") else link
    # TO DO: Remove Twitter documents as well
    for url in hrefs if url.startswith("https") and not url.startswith("https://www.youtube")
    for link in hrefs if not link.startswith("#")
    ]   
    return https_links


# https://github.com/langchain-ai/langchainjs/docs/get_started/quickstart/
def preprocess_document(url, index_name):
    embedding_model = VertexAIEmbeddings(project='arctic-acolyte-414610', model_name='textembedding-gecko@003')
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding_model)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 100,
        length_function = len,
        add_start_index = True)
    try:
        elements = partition_html(url=url)
        elements = chunk_by_title(elements)
        # elements = text_splitter.split_text(elements)
        # print("Elements", elements)
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
        vectorstore.add_documents(documents=documents)
        print("Doksh added")
    except Exception as e:
        print(e)
    # elements = partition_html(url=url)
    # element_dict = [el.to_dict() for el in elements]
    # example_output = json.dumps(element_dict, indent=2)
    # Convert string to an array
    # arr = ast.literal_eval(example_output)
    # print(example_output)



def load_data(url, index):
    internal_links = extract_internal_links(url)
    print(internal_links)

    with tqdm(total=len(internal_links)) as pbar:
        for link in internal_links:
            preprocess_document(link, index)
            pbar.update(1)

    return "Vector store updated successfully!"


print(
    load_data(
        "https://python.langchain.com/docs/get_started/introduction",
        "langchain-test-index",
    )
)
# print(load_data("https://python.langchain.com/docs/get_started/introduction","langchain-test-index" ))
# print(extract_internal_links("https://python.langchain.com/docs/get_started/introduction"))
# preprocess_document("https://python.langchain.com/docs/get_started/introduction", "langchain-test-index")