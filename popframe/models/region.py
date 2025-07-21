from functools import singledispatchmethod
import dill as pickle
import pyproj
import geopandas as gpd
import pandas as pd
from .town import Town
import matplotlib.pyplot as plt
from popframe.preprocessing.level_filler import LevelFiller
from popframe.utils.validators import RegionValidator
from popframe.config.constants import VISUALIZATION_COLORS


class Region():
    """
    A class representing a geographical region that includes districts, settlements, towns, and optionally territories.
    Provides methods for validating and visualizing spatial data, as well as for calculating accessibility between towns.

    Attributes
    ----------
    crs : pyproj.CRS
        Coordinate Reference System of the region.
    region : gpd.GeoDataFrame
        GeoDataFrame representing the boundaries of the region.
    districts : gpd.GeoDataFrame
        GeoDataFrame containing information about districts.
    settlements : gpd.GeoDataFrame
        GeoDataFrame containing information about settlements.
    _towns : dict
        Dictionary containing `Town` objects indexed by their IDs.
    accessibility_matrix : pd.DataFrame
        DataFrame containing accessibility data between towns.
    _territories : dict
        Dictionary containing `Territory` objects indexed by their IDs (optional).
    """

    def __init__(
            self, 
            region : gpd.GeoDataFrame, 
            towns : gpd.GeoDataFrame, 
            accessibility_matrix : pd.DataFrame, 
        ):
        """
        Initializes the Region object with GeoDataFrames for region, districts, settlements, and towns. 
        Optionally includes territories and an accessibility matrix to model transportation between towns.

        Parameters
        ----------
        region : gpd.GeoDataFrame
            GeoDataFrame representing the boundaries of the region.
        towns : gpd.GeoDataFrame
            GeoDataFrame containing information about towns.
        accessibility_matrix : pd.DataFrame
            DataFrame containing accessibility data between towns.

        Raises
        ------
        AssertionError
            If the CRS or indices between the towns and the accessibility matrix do not match.
        """
        # Используем новые валидаторы
        region = RegionValidator.validate_region_gdf(region)
        towns = RegionValidator.validate_towns_gdf(towns)
        accessibility_matrix = RegionValidator.validate_accessibility_matrix(accessibility_matrix, towns)

        assert (accessibility_matrix.index == towns.index).all(), "Accessibility matrix indices and towns indices don't match"
        assert region.crs == towns.crs, 'CRS should match everywhere'

        self.crs = towns.crs
        self.region = region
        self._towns = Town.from_gdf(towns)
        
        self.accessibility_matrix = accessibility_matrix
    
    @staticmethod
    def validate_towns(gdf : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Validates the towns GeoDataFrame.
        
        Deprecated: Use RegionValidator.validate_towns_gdf instead
        """
        import warnings
        warnings.warn(
            "validate_towns is deprecated. Use RegionValidator.validate_towns_gdf instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return RegionValidator.validate_towns_gdf(gdf)

    @staticmethod
    def validate_region(gdf : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Validates the region GeoDataFrame to ensure it has the correct structure and data types.
        
        Deprecated: Use RegionValidator.validate_region_gdf instead
        """
        import warnings
        warnings.warn(
            "validate_region is deprecated. Use RegionValidator.validate_region_gdf instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return RegionValidator.validate_region_gdf(gdf)
    
    @staticmethod
    def validate_accessibility_matrix(df : pd.DataFrame) -> pd.DataFrame:
        """
        Validates the accessibility matrix, ensuring it has non-negative float values 
        and matching row and column indices.
        
        Deprecated: Use RegionValidator.validate_accessibility_matrix instead
        """
        import warnings
        warnings.warn(
            "validate_accessibility_matrix is deprecated. Use RegionValidator.validate_accessibility_matrix instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Нужно создать фиктивный towns GDF для совместимости
        towns_index = pd.Index(df.index)
        fake_towns = gpd.GeoDataFrame(index=towns_index)
        return RegionValidator.validate_accessibility_matrix(df, fake_towns)

    @property
    def towns(self) -> list[Town]:
        """
        Returns a list of all towns in the region.

        Returns
        -------
        list[Town]
            List of Town objects.
        """
        return list(self._towns.values())
    
    def get_update_towns_gdf(self, update_df: pd.DataFrame | None = None):
        """
        Get updated towns GeoDataFrame with optional population updates
        
        Parameters
        ----------
        update_df : pd.DataFrame, optional
            DataFrame with population updates
            
        Returns
        -------
        gpd.GeoDataFrame
            Updated towns GeoDataFrame
        """
        gdf = self.get_towns_gdf()
        if update_df is not None:
            # Обновляем значения населения в gdf из update_df
            gdf.update(update_df[['population']])
            
            level_filler = LevelFiller(towns=gdf)
            gdf = level_filler.fill_levels()
        return gdf


    def get_towns_gdf(self) -> gpd.GeoDataFrame:
        """
        Returns a GeoDataFrame representing all towns in the region, including their relationships with settlements and districts.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with town data.
        """
        data = [town.to_dict() for town in self.towns]
        gdf = gpd.GeoDataFrame(data, crs=self.crs)
        gdf.set_index('id', inplace=True, drop=False)
        gdf = gdf.rename_axis(None)
        return gdf.fillna(0)

    @singledispatchmethod
    def __getitem__(self, arg):
        """
        Overloaded subscript operator to access a town or accessibility data based on the argument type.
        
        Parameters
        ----------
        arg : int or tuple
            Integer to access a town by its ID, or tuple to retrieve accessibility data between two towns.
        
        Raises
        ------
        NotImplementedError
            If the argument type is unsupported.
        """
        raise NotImplementedError(f"Can't access object with such argument type {type(arg)}")

    # Make city_model subscriptable, to access block via ID like city_model[123]
    @__getitem__.register(int)
    def _(self, town_id):
        if not town_id in self._towns:
            raise KeyError(f"Can't find town with such id: {town_id}")
        return self._towns[town_id]

    @__getitem__.register(tuple)
    def _(self, towns):
        (town_a, town_b) = towns
        if isinstance(town_a, Town):
            town_a = town_a.id
        if isinstance(town_b, Town):
            town_b = town_b.id
        return self.accessibility_matrix.loc[town_a, town_b]
    
    @staticmethod
    def from_pickle(file_path: str):
        """
        Load a Region object from a .pickle file.

        Parameters
        ----------
        file_path : str
            Path to the .pickle file.

        Returns
        -------
        Region
            The loaded Region object.
        """
        with open(file_path, "rb") as f:
            state = pickle.load(f)
        return state

    def to_pickle(self, file_path: str):
        """
        Save the Region object to a .pickle file.

        Parameters
        ----------
        file_path : str
            Path to the .pickle file.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)