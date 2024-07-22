from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from collections.abc import Iterable


class ClientBase(ABC):
    @abstractmethod
    def get_text_generation_parameters(
        self,
        **kwargs,
    ) -> dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def generate_text(
        self,
        model_id: str,
        text: str,
        **kwargs,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_text_stream(
        self,
        model_id: str,
        text: str,
        **kwargs,
    ) -> Iterable[str]:
        raise NotImplementedError

    @abstractmethod
    def embedding(
        self,
        model_id: str,
        text: str,
        timeout: float = 60.0,
        parameters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def embedding_tasks(
        self,
        model_id: str,
        texts: list[str],
        timeout: float = 60.0,
        parameters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def sentence_similarity(
        self,
        model_id: str,
        source_sentence: str,
        sentences: list[str],
        timeout: float = 60.0,
        parameters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def sentence_similarity_tasks(
        self,
        model_id: str,
        source_sentences: list[str],
        sentences: list[str],
        timeout: float = 60.0,
        parameters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def rerank(
        self,
        model_id: str,
        documents: list[dict[str, Any]],
        query: str,
        timeout: float = 60.0,
        parameters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def rerank_tasks(
        self,
        model_id: str,
        documents: list[dict[str, Any]],
        queries: list[str],
        timeout: float = 60.0,
        parameters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def models_info(self) -> list[dict[str, Any]]:
        raise NotImplementedError
