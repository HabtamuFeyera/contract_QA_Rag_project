from langchain_openai import OpenAIEmbeddings

class OpenAIEmbeddingsWrapper:
    def __init__(self, openai_api_key):
        """Initializes the OpenAIEmbeddingsWrapper with the OpenAI API key."""
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    def encode(self, document):
        """
        Generates embeddings for the provided document.

        Args:
            document (str): The document or text to encode.

        Returns:
            list: The generated embeddings.
        """
        # If using embed_documents, make sure to pass the document as a list
        return self.embeddings.embed_documents([document])[0]
