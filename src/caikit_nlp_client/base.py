from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


class ClientBase(ABC):
    @abstractmethod
    def get_text_generation_parameters(
        self,
        timeout: float,
        **kwargs,
    ) -> dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def generate_text(
        self,
        model_id: str,
        text: str,
        timeout: float,
        **kwargs,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_text_stream(
        self,
        model_id: str,
        text: str,
        timeout: float,
        **kwargs,
    ) -> Iterable[str]:
        raise NotImplementedError
