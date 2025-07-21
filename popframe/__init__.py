"""
PopFrame - библиотека для моделирования каркаса расселения и оценки территорий

PopFrame предоставляет инструменты для построения универсальной информационной модели 
региона на основе населённых пунктов, а также для сценарного моделирования и анализа.
"""

__version__ = "0.1.0"
__author__ = "IDU ITMO Team"
__email__ = "contact@idu.itmo.ru"
__credits__ = ["IDU ITMO", "National Center for Cognitive Research"]
__license__ = "BSD-3"

# Core exports - only the most essential classes to avoid circular imports
from .models.region import Region
from .models.town import Town

# Configuration - safe to import
from .config.constants import (
    AgglomerationConfig, 
    PopulationThresholds, 
    InfrastructureConfig,
    SpatialConfig,
    VISUALIZATION_COLORS,
    UNITS
)

__all__ = [
    # Models
    "Region", 
    "Town",
    
    # Configuration
    "AgglomerationConfig",
    "PopulationThresholds",
    "InfrastructureConfig", 
    "SpatialConfig",
    "VISUALIZATION_COLORS",
    "UNITS",
    
    # Meta
    "__version__",
]

# Lazy imports for analysis methods to avoid circular dependencies
def __getattr__(name):
    """Lazy import of analysis methods"""
    if name == "AgglomerationBuilder":
        from .method.agglomeration import AgglomerationBuilder
        return AgglomerationBuilder
    elif name == "InfrastructureAnalyzer":
        from .method.engineer import InfrastructureAnalyzer
        return InfrastructureAnalyzer
    elif name == "CityPopulationScorer":
        from .method.city_evaluation import CityPopulationScorer
        return CityPopulationScorer
    elif name == "SpatialInequalityCalculator":
        from .method.spatial_inequality import SpatialInequalityCalculator
        return SpatialInequalityCalculator
    elif name == "TerritoryEvaluation":
        from .method.territory_evaluation import TerritoryEvaluation
        return TerritoryEvaluation
    elif name == "LandUseAssessment":
        from .method.landuse_assessment import LandUseAssessment
        return LandUseAssessment
    elif name == "AnchorSettlementBuilder":
        from .method.anchor_settlement import AnchorSettlementBuilder
        return AnchorSettlementBuilder
    elif name == "LevelFiller":
        from .preprocessing.level_filler import LevelFiller
        return LevelFiller
    elif name == "PopulationFiller":
        from .preprocessing.population_filler import PopulationFiller
        return PopulationFiller
    elif name == "AdjacencyCalculator":
        from .preprocessing.adjacency_calculator import AdjacencyCalculator
        return AdjacencyCalculator
    elif name == "RegionValidator":
        from .utils.validators import RegionValidator
        return RegionValidator
    elif name == "DataValidator":
        from .utils.validators import DataValidator
        return DataValidator
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
