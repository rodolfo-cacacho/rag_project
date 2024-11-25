    
from pinecone import Pinecone, ServerlessSpec
from collections import Counter
from sentence_transformers import SentenceTransformer


class PineconeDBConnector:
    def __init__(self, api_key, index_name, embedding_model_name="aari1995/German_Semantic_V3", dimension=768, cloud="aws", region="us-east-1"):
        # Initialize the embedding model using SentenceTransformer
        self.embedding_model = SentenceTransformer(embedding_model_name, trust_remote_code=True)
        
        # Initialize Pinecone with the provided API key
        self.pc = Pinecone(api_key=api_key)
        self.dims = dimension
        self.cloud = cloud
        self.region = region
        
        print(f'Embedding Dimension: {dimension}')
        # Check if the index exists, if not, create a new one with ServerlessSpec
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=self.dims,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )
            print(f"Created new Pinecone index: {index_name}")
        else:
            print(f"Using existing Pinecone index: {index_name}")
        
        self.index_name = index_name
        self.index = self.pc.Index(index_name)

    def create_embedding(self, text: str):
        """
        Generate an embedding for a given text using SentenceTransformer.
        """
        return self.embedding_model.encode(text).tolist()
    
    def chunker(self,seq, size):
        """Yield successive chunks from a sequence."""
        for pos in range(0, len(seq), size):
            yield seq[pos:pos + size]

    def add_documents(self, splits_lists, ids, batch_size=500):
        """
        Add documents to the Pinecone index in smaller batches, calculating embeddings with SentenceTransformer.
        - splits_lists: List of document objects with content and metadata.
        - ids: List of unique document IDs.
        - batch_size: Number of documents to upsert at once.
        """
        vectors_to_upsert = []
        for index, x in enumerate(splits_lists):
            current_ids = ids[index]
            # Generate embeddings for the documents
            embeddings = [self.create_embedding(doc.page_content) for doc in x]
            metadatas = [doc.metadata for doc in x]

            for i, embedding in enumerate(embeddings):
                vectors_to_upsert.append({'id':current_ids[i],"values": embedding,"metadata": metadatas[i]})

        # Split vectors into batches and upsert them in chunks
        for chunk in self.chunker(vectors_to_upsert, batch_size):
            self.index.upsert(vectors=chunk)
            print(f"Upserted {len(chunk)} vectors to Pinecone.")

    def count_registers(self):
        """
        Count the number of vectors/documents in the Pinecone index.
        """
        stats = self.index.describe_index_stats()
        return stats['total_vector_count']

    def get_embeddings_by_ids(self, ids):
        """
        Retrieve embeddings for specific document IDs from the Pinecone index.
        """
        results = self.index.fetch(ids=ids)
        return [vector['values'] for vector in results['vectors'].values()]

    def reset_collection(self):
        """
        Reset the Pinecone index by deleting it and recreating it.
        """
        self.pc.delete_index(self.index_name)
        self.pc.create_index(
            name=self.index_name,
            dimension=self.dims,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=self.cloud,
                region=self.region
            )
        )
        self.index = self.pc.Index(self.index_name)
        print(f"Reset the Pinecone index: {self.index_name}")

    def query_collection(self, query_text: str, n_results: int = 5, include_metadata=True,where = None):
        """
        Query the Pinecone index using a provided text, generating the embedding for it.
        - query_text: The text to query.
        - n_results: Number of results to return.
        """
        query_embedding = self.create_embedding(query_text)
        query_result = self.index.query(
            vector=query_embedding,
            top_k=n_results,
            include_values=False,  # Return only the metadata and distances
            include_metadata=include_metadata,
            filter = where
        )
        return query_result