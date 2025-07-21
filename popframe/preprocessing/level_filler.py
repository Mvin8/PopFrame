"""
Level filling utilities for PopFrame
"""

from typing import ClassVar
from pydantic import BaseModel, Field, field_validator

from ..models.geodataframe import BaseRow, GeoDataFrame
from ..config.constants import PopulationThresholds


class TownRow(BaseRow):
    """
    A data class representing a town row with ID, name, population, and level.

    Attributes
    ----------
    id : int
        Unique identifier for the town.
    name : str
        Name of the town.
    population : int
        Population of the town. Must be greater than zero.
    level : str
        Administrative level of the town. Defaults to "Нет уровня".

    Methods
    -------
    None
    """

    id: int
    name: str
    population: int = Field(gt=0)
    level: str = Field(default="Нет уровня")


class LevelFiller(BaseModel):
    """
    A class for automatically assigning administrative levels to towns based on population thresholds.

    Attributes
    ----------
    towns : GeoDataFrame[TownRow]
        A GeoDataFrame containing town data that includes name, population, and level.
    population_thresholds : PopulationThresholds
        Configuration object defining population ranges for different administrative levels.

    Methods
    -------
    _assign_level(row) -> str
        A static method that assigns the correct administrative level to a town based on its population.
        
    validate_towns(gdf)
        A Pydantic validator that ensures town levels are correctly assigned before processing the GeoDataFrame.

    fill_levels() -> GeoDataFrame[TownRow]
        Fills in the levels for all towns in the GeoDataFrame based on population and returns the updated GeoDataFrame.
    """

    towns: GeoDataFrame[TownRow]
    population_thresholds: PopulationThresholds = Field(default_factory=PopulationThresholds)

    @field_validator("towns", mode="before")
    def validate_towns(cls, gdf):
        """
        Validates the towns GeoDataFrame and assigns levels based on population.
        
        Parameters
        ----------
        gdf : GeoDataFrame
            The input GeoDataFrame to validate
            
        Returns
        -------
        GeoDataFrame
            The validated and processed GeoDataFrame
        """
        if not isinstance(gdf, GeoDataFrame[TownRow]):
            gdf = GeoDataFrame[TownRow](gdf)
            
        # Assign levels based on population
        thresholds = PopulationThresholds()
        gdf['level'] = gdf.apply(lambda row: cls._assign_level(row, thresholds), axis=1)
        
        return gdf

    @staticmethod
    def _assign_level(row, thresholds: PopulationThresholds) -> str:
        """
        Assigns the administrative level based on population thresholds.

        Parameters
        ----------
        row : pandas.Series
            A row from the towns GeoDataFrame containing population data.
        thresholds : PopulationThresholds
            Configuration object with population thresholds

        Returns
        -------
        str
            The administrative level corresponding to the population.
        """
        population = row.get('population', 0)
        
        for level, (min_pop, max_pop) in thresholds.THRESHOLDS.items():
            if min_pop <= population < max_pop:
                return level
                
        return "Нет уровня"

    def fill_levels(self) -> GeoDataFrame[TownRow]:
        """
        Fills in the administrative levels for all towns in the GeoDataFrame based on their population.

        Returns
        -------
        GeoDataFrame[TownRow]
            The GeoDataFrame with updated administrative levels.
        """
        # Levels are already assigned in the validator, so we just return the towns
        return self.towns

