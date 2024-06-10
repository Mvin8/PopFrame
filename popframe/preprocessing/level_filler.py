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
    population_thresholds: ClassVar[dict[str, int]] = {
        "Сверхкрупный город": 3000000,
        "Крупнейший город": 1000000,
        "Крупный город": 250000,
        "Большой город": 100000,
        "Средний город": 50000,
        "Малый город": 50000,
        "Крупное сельское поселение": 5000,
        "Большое сельское поселение": 1000,
        "Среднее сельское поселение": 200,
        "Малое сельское поселение": 200,
    }

    @staticmethod
    def _assign_level(row) -> str:
        population = row['population']
        
        if population > LevelFiller.population_thresholds["Сверхкрупный город"]:
            return "Сверхкрупный город"
        elif population > LevelFiller.population_thresholds["Крупнейший город"]:
            return "Крупнейший город"
        elif population > LevelFiller.population_thresholds["Крупный город"]:
            return "Крупный город"
        elif population > LevelFiller.population_thresholds["Большой город"]:
            return "Большой город"
        elif population > LevelFiller.population_thresholds["Средний город"]:
            return "Средний город"
        elif population > LevelFiller.population_thresholds["Крупное сельское поселение"]:
            return "Малый город"
        elif population > LevelFiller.population_thresholds["Большое сельское поселение"]:
            return "Крупное сельское поселение"
        elif population > LevelFiller.population_thresholds["Среднее сельское поселение"]:
            return "Большое сельское поселение"
        elif population > LevelFiller.population_thresholds["Малое сельское поселение"]:
            return "Среднее сельское поселение"
        else:
            return "Малое сельское поселение"

    @field_validator("towns", mode="before")
    @classmethod
    def validate_towns(cls, gdf):
        gdf["level"] = gdf.apply(cls._assign_level, axis=1)
        return GeoDataFrame[TownRow](gdf)

    def fill_levels(self) -> gpd.GeoDataFrame:
        towns = self.towns.copy()
        towns["level"] = towns.apply(self._assign_level, axis=1)
        return towns
