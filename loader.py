from dotenv import load_dotenv
from tqdm import tqdm
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Pinecone as PineconeLangChain
# from pinecone import Pinecone
from unstructured.partition.html import partition_html
from unstructured.chunking.basic import chunk_elements
from langchain_core.documents import Document
import os
import nltk
nltk.download('averaged_perceptron_tagger')


load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "arctic-acolyte-414610-c6dcb23dd443.json"


def extract_internal_links(url):
    elements = partition_html(url=url)
    links = []

    for element in elements:
        if element.metadata.link_urls:
            relative_link = element.metadata.link_urls[0][1:]
            if relative_link.startswith("v0.2/docs"):
                links.append(f"https://python.langchain.com/{relative_link}")
                url = "https://python.langchain.com/" + relative_link
                partitions = partition_html(url=url)
                for partition in partitions:
                    if partition.metadata.link_urls:
                        relative_link = partition.metadata.link_urls[0][1:]
                        if relative_link.startswith("v0.2/docs"):
                            links.append(f"https://python.langchain.com/{relative_link}")
    print(links)
    return links


# https://github.com/langchain-ai/langchainjs/docs/get_started/quickstart/
def preprocess_document(url, index_name):
    embedding_model = VertexAIEmbeddings(project='arctic-acolyte-414610', model_name='textembedding-gecko@003')
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding_model)
    # delete existing contet
    # delete_response = vectorstore.delete(delete_all=True)
    # print(delete_response)
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
        vectorstore.add_documents(documents=documents)
        print("Doksh added")
    except Exception as e:
        print(e)


def load_data(url, index):
    internal_links = extract_internal_links(url)
    embedding_model = VertexAIEmbeddings(project='arctic-acolyte-414610', model_name='textembedding-gecko@003')
    vectorstore = PineconeVectorStore(index_name='langchain-test-index', embedding=embedding_model)
    # delete existing contet
    # delete_response = vectorstore.delete(delete_all=True)
    # print(delete_response)

    with tqdm(total=len(internal_links)) as pbar:
        for link in internal_links:
            preprocess_document(link, index)
            pbar.update(1)

    return "Vector store updated successfully!"


print(
    load_data(
        "https://python.langchain.com/v0.2/docs/introduction/",
        "langchain-test-index",
    )
)
# print(load_data("https://python.langchain.com/docs/get_started/introduction","langchain-test-index" ))
# print(extract_internal_links("https://python.langchain.com/v0.2/docs/introduction/"))
# preprocess_document("https://python.langchain.com/docs/get_started/introduction", "langchain-test-index")