"""
Base classes for PopFrame analysis methods
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, InstanceOf

from ..models.region import Region


class BaseMethod(ABC, BaseModel):
    """Базовый класс для всех методов анализа, работающих с регионом"""

    region: InstanceOf[Region]

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """
        Основной метод выполнения анализа
        
        Parameters
        ----------
        **kwargs
            Параметры анализа
            
        Returns
        -------
        Any
            Результат анализа
        """
        pass
    
    def validate_input(self, **kwargs) -> bool:
        """
        Валидация входных параметров
        
        Parameters
        ----------
        **kwargs
            Параметры для валидации
            
        Returns
        -------
        bool
            True если валидация прошла успешно
            
        Raises
        ------
        ValueError
            При некорректных входных данных
        """
        return True


class BaseAnalyzer(ABC):
    """Базовый класс для анализаторов без привязки к Region"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация анализатора
        
        Parameters
        ----------
        config : dict, optional
            Конфигурация анализатора
        """
        self.config = config or {}
    
    @abstractmethod
    def analyze(self, **kwargs) -> Any:
        """
        Выполнение анализа
        
        Parameters
        ----------
        **kwargs
            Параметры анализа
            
        Returns
        -------
        Any
            Результат анализа
        """
        pass
    
    def validate_input(self, **kwargs) -> bool:
        """
        Валидация входных параметров
        
        Parameters
        ----------
        **kwargs
            Параметры для валидации
            
        Returns
        -------
        bool
            True если валидация прошла успешно
        """
        return True
