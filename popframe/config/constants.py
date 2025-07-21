"""
Configuration classes and constants for PopFrame methods
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class AgglomerationConfig:
    """Конфигурация для построения агломераций"""
    
    DEFAULT_RADIUS: int = 400
    MIN_POPULATION: int = 15000
    LEVEL_TIME_REDUCTION: int = 10
    MIN_TIME_THRESHOLD: int = 50
    MAX_TIME_THRESHOLD: int = 120
    
    CITY_LEVELS: List[str] = None
    
    def __post_init__(self):
        if self.CITY_LEVELS is None:
            self.CITY_LEVELS = [
                "Малый город",
                "Средний город",
                "Большой город", 
                "Крупный город",
                "Крупнейший город",
                "Сверхкрупный город"
            ]


@dataclass
class PopulationThresholds:
    """Пороги численности населения для классификации поселений"""
    
    THRESHOLDS: Dict[str, Tuple[int, int]] = None
    
    def __post_init__(self):
        if self.THRESHOLDS is None:
            self.THRESHOLDS = {
                "Сверхкрупный город": (3000000, float('inf')),
                "Крупнейший город": (1000000, 3000000),
                "Крупный город": (250000, 1000000),
                "Большой город": (100000, 250000),
                "Средний город": (50000, 100000),
                "Малый город": (5000, 50000),
                "Крупное сельское поселение": (3000, 5000),
                "Большое сельское поселение": (1000, 3000),
                "Среднее сельское поселение": (200, 1000),
                "Малое сельское поселение": (0, 200),
            }


@dataclass 
class InfrastructureConfig:
    """Конфигурация для анализа инфраструктуры"""
    
    DEFAULT_RADIUS: float = 1000.0
    NUCLEAR_PLANT_RADIUS: float = 100000.0
    HYDRO_PLANT_RADIUS: float = 10000.0
    
    NUCLEAR_KEYWORDS: List[str] = None
    HYDRO_KEYWORDS: List[str] = None
    
    def __post_init__(self):
        if self.NUCLEAR_KEYWORDS is None:
            self.NUCLEAR_KEYWORDS = ["Атомная электростанция", "АЭС"]
        if self.HYDRO_KEYWORDS is None:
            self.HYDRO_KEYWORDS = ["Гидроэлектростанция", "ГЭС"]


@dataclass
class SpatialConfig:
    """Конфигурация для пространственных операций"""
    
    DEFAULT_CRS: int = 4326
    METRIC_CRS: int = 3857
    MIN_AREA_THRESHOLD: float = 1.0  # кв. метры
    BUFFER_RESOLUTION: int = 16


# Константы визуализации
VISUALIZATION_COLORS = {
    'districts': '#28486d',
    'settlements': '#ddd', 
    'towns': '#333333',
    'territories': '#893434',
    'agglomerations': '#ff6b6b'
}

# Константы единиц измерения
UNITS = {
    'square_meters_in_hectare': 10_000,
    'meters_in_km': 1_000
}