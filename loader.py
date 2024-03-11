from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from tqdm import tqdm
import pinecone
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from pinecone import Pinecone
import requests
from bs4 import BeautifulSoup
import urllib


load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="arctic-acolyte-414610-c6dcb23dd443.json"

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
    for url in hrefs if url.startswith("https")
    for link in hrefs if not link.startswith("#")
    ]   
    return https_links

def load_data(url, index_name):
    
    internal_links = extract_internal_links(url)
    print(internal_links)
    
    embeddings = VertexAIEmbeddings(project='arctic-acolyte-414610', model_name='textembedding-gecko@003')
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 100,
        length_function = len,
        add_start_index = True)
    
    with tqdm(total=len(internal_links)) as pbar:
        for link in internal_links:
            try:
                loader = WebBaseLoader(link)
                data = loader.load()
                docs = text_splitter.split_documents(data)
                vectorstore.add_documents(docs)
            except Exception as e:
                print(e)
            pbar.update(1)

    return "Done"

print(load_data("https://python.langchain.com/docs/get_started/introduction","langchain-test-index" ))
# print(extract_internal_links("https://python.langchain.com/docs/get_started/introduction"))