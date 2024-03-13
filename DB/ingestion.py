import os
import pinecone
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm.auto import tqdm


load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "arctic-acolyte-414610-c6dcb23dd443.json"

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment="gcp-starter",
)
INDEX_NAME = "langchain-doc-index"
index = pinecone.Index(INDEX_NAME)


def ingest_docs():
    """Ingesting documents from any source, converting them to text embeddings, and storing the embeddings
    into the vector store. This opertion should be done only once for each data source, or if you
    need to update the data stored in the vector store"""

    dataset = load_dataset("jamescalam/langchain-docs", split="train")
    embeddings = VertexAIEmbeddings(project="favorable-beach-405907")
    data = dataset.to_pandas()

    batch_size = 100

    for i in tqdm(range(1700, 2000, batch_size)):
        i_end = min(len(data), i + batch_size)
        # get batch of data
        batch = data.iloc[i:i_end]
        # generate unique ids for each chunk
        ids = [f"{x['id']}" for i, x in batch.iterrows()]
        # get text to embed
        texts = [x["text"] for _, x in batch.iterrows()]
        # embed text
        embeds = embeddings.embed_documents(texts)
        # get metadata to store in Pinecone
        metadata = [
            {"text": x["text"], "source": x["source"]} for i, x in batch.iterrows()
        ]
        # add to Pinecone
        index.upsert(vectors=zip(ids, embeds, metadata))


if __name__ == "__main__":
    ingest_docs()
