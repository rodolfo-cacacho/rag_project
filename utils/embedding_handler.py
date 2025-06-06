class EmbeddingHandler:
    def __init__(self, model_name=None, use_api=False, api_function=None, task=None):
        """
        Initializes the embedding handler.

        Args:
            model_name (str): Name of the transformer model to load for local embedding.
            use_api (bool): Whether to use an API for embeddings.
            api_function (function): Function to call for API-based embedding.
            task (str): Task to specify for models supporting tasks like "retrieval.query".
        """
        self.use_api = use_api
        self.api_function = api_function
        self.embedding_model = None
        self.model_name = model_name
        self.task = task

        if not use_api and model_name:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(model_name, trust_remote_code=True)
            self.model_dimension = self.embedding_model.get_sentence_embedding_dimension()
        elif use_api and api_function:
            self.model_name = "API"
            self.model_dimension = None  # Can set this dynamically if API provides it.

    def get_detailed_instruct(self, task_description, query):
        """
        Formats the instruction and query for models that require a specific input structure.

        Args:
            task_description (str): The task description to prepend.
            query (str): The actual query text.

        Returns:
            str: Formatted instruction and query.
        """
        return f'Instruct: {task_description}\nQuery: {query}'

    def embed_texts(self, texts, task=None, instruction=None,normalize = False):
        """
        Embeds texts using the specified method (API or transformer model).
        
        Args:
            texts (list): List of strings to embed.
            task (str): Task to specify for models supporting tasks like "retrieval.query".
            instruction (str): Optional instruction to prepend to each query.

        Returns:
            list: List of embeddings, with all values as Python floats.
        """
        if self.use_api:
            if not self.api_function:
                raise ValueError("API function must be defined for API embedding.")
            
            # Add instruction if provided
            if instruction:
                texts = [self.get_detailed_instruct(instruction, text) for text in texts]
            
            return [[float(value) for value in self.api_function(text)] for text in texts]

        if not self.embedding_model:
            raise ValueError("Transformer model must be loaded for local embedding.")
        
        # Use task from the class or the passed argument
        task_to_use = task or self.task

        # Add instruction if provided
        if instruction:
            texts = [self.get_detailed_instruct(instruction, text) for text in texts]

        if task_to_use:
            # Encode with task-specific arguments
            embeddings = self.embedding_model.encode(
                texts,
                task=task_to_use,
                prompt_name=task_to_use,
                batch_size=16,
                show_progress_bar=False,
                normalize_embeddings = normalize
            )
        else:
            # Default encoding without task
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=16,
                normalize_embeddings = normalize,
                show_progress_bar=False
            )

        # Ensure all embedding values are converted to Python floats
        return [[float(value) for value in embedding] for embedding in embeddings]