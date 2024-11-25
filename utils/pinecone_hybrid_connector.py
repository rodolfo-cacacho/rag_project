from pinecone import Pinecone, ServerlessSpec

class PineconeDBConnectorHybrid:
    def __init__(self, api_key, index_name, dimension=768, cloud="aws", region="us-east-1", metric="dotproduct"):
        """
        Initialize the PineconeDBConnectorHybrid class.
        """
        # Initialize Pinecone with the provided API key
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dims = dimension
        self.cloud = cloud
        self.region = region
        self.metric = metric

        print(f"Embedding Dimension: {dimension}")

        # Check if the index exists, otherwise create it
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=self.dims,
                metric=self.metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            print(f"Created new Pinecone index: {index_name}")
        else:
            print(f"Using existing Pinecone index: {index_name}")

        self.index = self.pc.Index(index_name)

    def upsert_vectors(self, vectors, batch_size=100):
        """
        Upserts vectors to Pinecone in batches.

        Args:
            vectors (list): List of vectors, each a dictionary with 'id', 'values', and optional 'sparse_values'.
            batch_size (int): Number of vectors to upload in each batch.
        """
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            print(f"Uploaded batch {i // batch_size + 1}/{(len(vectors) + batch_size - 1) // batch_size}")

    def query_collection(self, query_embedding_dense, query_embedding_sparse=None, n_results=5, include_metadata=True, where=None):
        """
        Query the Pinecone index using the provided embeddings.

        Args:
            query_embedding_dense (list): Dense embedding vector for the query.
            query_embedding_sparse (dict): Sparse embedding vector for the query (optional).
            n_results (int): Number of top results to return.
            include_metadata (bool): Whether to include metadata in the results.
            where (dict): Optional filter for metadata-based search.

        Returns:
            dict: Query results from Pinecone.
        """
        query_result = self.index.query(
            vector=query_embedding_dense,
            sparse_vector=query_embedding_sparse,
            top_k=n_results,
            include_metadata=include_metadata,
            filter=where,
        )
        return query_result

    def count_registers(self):
        """
        Count the number of vectors/documents in the Pinecone index.

        Returns:
            int: Total number of vectors in the index.
        """
        stats = self.index.describe_index_stats()
        return stats['total_vector_count']

    def reset_collection(self):
        """
        Resets the Pinecone index by deleting it and recreating it.

        WARNING: This will delete all existing data in the index.
        """
        self.pc.delete_index(self.index_name)
        self.pc.create_index(
            name=self.index_name,
            dimension=self.dims,
            metric=self.metric,
            spec=ServerlessSpec(cloud=self.cloud, region=self.region),
        )
        self.index = self.pc.Index(self.index_name)
        print(f"Reset the Pinecone index: {self.index_name}")

