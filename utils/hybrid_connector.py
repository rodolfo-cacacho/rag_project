from pinecone import Pinecone, ServerlessSpec
from collections import Counter


class PineconeDBConnectorHybrid:
    def __init__(self, api_key, index_name, dimension=768, cloud="aws", region="us-east-1", metric = "dotproduct"):
        
        # Initialize Pinecone with the provided API key
        self.pc = Pinecone(api_key=api_key)
        self.dims = dimension
        self.cloud = cloud
        self.region = region
        self.metric = metric
        
        print(f'Embedding Dimension: {dimension}')
        # Check if the index exists, if not, create a new one with ServerlessSpec
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=self.dims,
                metric=self.metric,
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


    def build_dict(self,input_batch):
    # store a batch of sparse embeddings
        sparse_emb = []
        # iterate through input batch
        for token_ids in input_batch:
            # convert the input_ids list to a dictionary of key to frequency values
            d = dict(Counter(token_ids))
            # remove special tokens and append sparse vectors to sparse_emb list
            sparse_emb.append({key: d[key] for key in d if key not in self.special_tokens})
        # return sparse_emb list
        return sparse_emb

    def generate_sparse_vectors(self,context_batch):
        # create batch of input_ids
        inputs = self.tokenizer(
                context_batch, padding=True,
                truncation=True,
                max_length=512
        )['input_ids']
        # create sparse dictionaries
        sparse_embeds = self.build_dict(inputs)
        return sparse_embeds
    
    def chunker(self,seq, size):
        """Yield successive chunks from a sequence."""
        for pos in range(0, len(seq), size):
            yield seq[pos:pos + size]

    # Editar function para subir los documentos con Sparse embeddings
    def add_documents(self, doc_lists, ids, batch_size=500, print_t=1000):
        """
        Add documents to the Pinecone index in smaller batches, calculating embeddings with SentenceTransformer.
        - doc_lists: List of document objects with content and metadata.
        - ids: List of unique document IDs.
        - batch_size: Number of documents to upsert at once.
        """
        vectors_to_upsert = []
        total_docs = len(doc_lists)

        for start_idx in range(0, total_docs, batch_size):
            end_idx = min(start_idx + batch_size, total_docs)
            batch_docs = doc_lists[start_idx:end_idx]
            batch_ids = ids[start_idx:end_idx]

            # Generate dense embeddings in chunks
            embeddings_dense = [self.create_dense_embedding(doc['content']) for doc in batch_docs]
            metadatas = [doc['metadata'] for doc in batch_docs]
            embeddings_sparse = [doc['sparse_values'] for doc in batch_docs]

            # Prepare the batch to upsert
            batch_vectors = []
            for i, embedding in enumerate(embeddings_dense):
                vector = {
                    'id': batch_ids[i],
                    'values': embedding,
                    'metadata': metadatas[i],
                }
                if len(embeddings_sparse[i]['indices']) > 0:
                    vector['sparse_values'] = embeddings_sparse[i]
                batch_vectors.append(vector)

            # Append to the main list and upsert in chunks
            vectors_to_upsert.extend(batch_vectors)
            if (end_idx) % print_t == 0:
                print(f'{end_idx}/{total_docs} vectors created.')

            # Upsert in chunks
            while len(vectors_to_upsert) >= batch_size:
                chunk = vectors_to_upsert[:batch_size]
                self.index.upsert(vectors=chunk)
                vectors_to_upsert = vectors_to_upsert[batch_size:]
                print(f"Upserted {len(chunk)} vectors to Pinecone.")

        # Upsert any remaining vectors
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert)
            print(f"Upserted final {len(vectors_to_upsert)} vectors to Pinecone.")

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
            metric=self.metric,
            spec=ServerlessSpec(
                cloud=self.cloud,
                region=self.region
            )
        )
        self.index = self.pc.Index(self.index_name)
        print(f"Reset the Pinecone index: {self.index_name}")

    def query_collection(self, query_text: str,sparse_embedding, n_results: int = 5, include_metadata=True,where = None,alpha = 0.8):
        """
        Query the Pinecone index using a provided text, generating the embedding for it.
        - query_text: The text to query.
        - n_results: Number of results to return.
        """
        query_embedding_dense = self.create_dense_embedding(query_text)
        query_embedding_sparse = sparse_embedding

        # scale alpha with hybrid_scale
        dense_vec, sparse_vec = self.hybrid_scale(
            query_embedding_dense, query_embedding_sparse, alpha
        )

        query_result = self.index.query(
            vector=dense_vec,
            top_k=n_results,
            sparse_vec = sparse_vec,
            include_values=False,  # Return only the metadata and distances
            include_metadata=include_metadata,
            filter = where
        )
        return query_result
    
    def hybrid_scale(self,dense, sparse, alpha: float = 1.0):
        # check alpha value is in range
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")
        # scale sparse and dense vectors to create hybrid search vecs
        hsparse = {
            'indices': list(sparse.keys()),
            'values':  [v * (1 - alpha) for v in sparse.values()]
        }
        hdense = [v * alpha for v in dense]
        return hdense, hsparse
    
    def upsert_vectors(self, vectors, batch_size=100):
        """
        Upserts vectors to Pinecone in batches to handle payload limits.
        
        Args:
            vectors (list): List of vectors, each a dictionary with 'id', 'values', and optional 'sparse_values'.
            batch_size (int): Number of vectors to upload in each batch.
        """
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            # print(f"Uploaded batch {i // batch_size + 1}/{(len(vectors) + batch_size - 1) // batch_size}")

