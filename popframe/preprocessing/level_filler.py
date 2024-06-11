import geopandas as gpd
from pydantic import BaseModel, Field, field_validator
from shapely.geometry import Point
from typing import ClassVar
from ..models.geodataframe import GeoDataFrame, BaseRow


class TownRow(BaseRow):
    geometry: Point
    name: str
    population : int = Field(gt=0)
    level: str = Field(default="Нет уровня")


class LevelFiller(BaseModel):
    towns: GeoDataFrame[TownRow]
    population_thresholds: ClassVar[dict[str, tuple[int, int]]] = {
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

    @staticmethod
    def _assign_level(row) -> str:
        population = row['population']
        
        for level, (lower_bound, upper_bound) in LevelFiller.population_thresholds.items():
            if lower_bound < population <= upper_bound:
                return level
        return "Неизвестный уровень"  # На случай, если население не вписывается ни в один из диапазонов

    @field_validator("towns", mode="before")
    @classmethod
    def validate_towns(cls, gdf):
        gdf["level"] = gdf.apply(cls._assign_level, axis=1)
        return GeoDataFrame[TownRow](gdf)

    def fill_levels(self) -> GeoDataFrame[TownRow]:
        towns = self.towns.copy()
        towns["level"] = towns.apply(self._assign_level, axis=1)
        return towns
