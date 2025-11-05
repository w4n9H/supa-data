from supadata.core import AutoLLM
from supadata.rag.long_context_rag import LongContextRAG
from supadata.supatypes import SupaDataArgs


class RAGFactory:
    @staticmethod
    def get_rag(llm: AutoLLM, args: SupaDataArgs, path: str, **kargs) -> LongContextRAG:
        """
        Factory method to get the appropriate RAG implementation based on arguments.
        Args:
            llm (AutoLLM): The ByzerLLM instance.
            args (AutoCoderArgs): The arguments for configuring RAG.
            path (str): The path to the data.
        Returns:
            SimpleRAG or LongContextRAG: The appropriate RAG implementation.
        """
        return LongContextRAG(llm, args, path, **kargs)