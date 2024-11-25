import chromadb

class ChromaDBConnector:
    def __init__(self, storage_path, collection_name, embedding_model_name="aari1995/German_Semantic_V3"):
        # Initialize the embedding model using SentenceTransformer
        self.embedding_model = SentenceTransformer(embedding_model_name,trust_remote_code = True)
        
        # Initialize ChromaDB Persistent Client
        self.client = chromadb.PersistentClient(path=storage_path)
        self.collection_name = collection_name
        self.col = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """
        Create or retrieve a collection from ChromaDB.
        """
        try:
            # Try to get an existing collection
            col = self.client.get_collection(name=self.collection_name)
            print(f'Collection {self.collection_name} already exists with {col.count()} documents.')
        except ValueError:
            # If the collection doesn't exist, create it
            col = self.client.get_or_create_collection(self.collection_name)
            print(f'Created new collection {self.collection_name}.')
        return col

    def create_embedding(self, text: str):
        """
        Generate an embedding for a given text using SentenceTransformer.
        """
        return self.embedding_model.encode(text).tolist()

    def add_documents(self, splits_lists, ids):
        """
        Add documents to the collection, calculating embeddings with SentenceTransformer.
        - splits_lists: List of document objects with content and metadata.
        - ids: List of unique document IDs.
        """
        created_ids = []
        for index, x in enumerate(splits_lists):
            current_ids = ids[index]
            # Generate embeddings for the documents
            embeddings = [self.create_embedding(doc.page_content) for doc in x]

            self.col.add(
                ids=current_ids,
                documents=[doc.page_content for doc in x],
                metadatas=[doc.metadata for doc in x],
                embeddings=embeddings  # Pass the pre-calculated embeddings
            )
            created_ids.extend(current_ids)
            print(f'Actual registers {self.col.count()}')
        return created_ids

    def count_registers(self):
        """
        Count the number of documents in the collection.
        """
        return self.col.count()
    
    def get_embeddings_by_ids(self, ids):
        """
        Retrieve embeddings for specific document IDs from the collection.
        """
        results = self.col.get(ids=ids, include=['embeddings'])
        return results['embeddings']

    def reset_collection(self):
        """
        Reset the collection by deleting and recreating it.
        """
        self.client.delete_collection(name=self.collection_name)
        self.col = self._get_or_create_collection()
        print(f'Collection {self.collection_name} has been reset.')

    def query_collection(self, query_text: str, n_results: int = 5,include = ['distances'],where = None):
        """
        Query the collection using a provided text, generating the embedding for it.
        - query_text: The text to query.
        - n_results: Number of results to return.
        """
        query_embedding = self.create_embedding(query_text)
        results = self.col.query(query_embeddings=[query_embedding], n_results=n_results,include=include,where = where)
        return results