from functools import singledispatchmethod
import dill as pickle
import pyproj
import shapely
import geopandas as gpd
import pandas as pd
from .town import Town
from .territory import Territory
import matplotlib.pyplot as plt


DISTRICTS_PLOT_COLOR = '#28486d'
SETTLEMENTS_PLOT_COLOR = '#ddd'
TOWNS_PLOT_COLOR = '#333333'
TERRITORIES_PLOT_COLOR = '#893434'

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

    Methods
    -------
    validate_districts(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame
        Validates the structure and content of a districts GeoDataFrame.
        
    validate_settlements(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame
        Validates the structure and content of a settlements GeoDataFrame.
        
    validate_towns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame
        Validates the structure and content of a towns GeoDataFrame.
        
    validate_region(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame
        Validates the structure and content of a region GeoDataFrame.
        
    validate_accessibility_matrix(df: pd.DataFrame) -> pd.DataFrame
        Validates the structure and content of the accessibility matrix.
    
    plot(figsize=(15, 15))
        Visualizes the region, including districts, settlements, towns, and optionally territories.

    get_territory(territory_id: int) -> Territory
        Retrieves a territory by its ID.
    
    towns() -> list[Town]
        Returns a list of all towns in the region.
    
    territories() -> list[Territory]
        Returns a list of all territories in the region.

    geometry() -> shapely.Polygon | shapely.MultiPolygon
        Returns the unified geometry of all districts in the region.
    
    get_territories_gdf() -> gpd.GeoDataFrame
        Returns a GeoDataFrame representing all territories in the region.
    
    get_towns_gdf() -> gpd.GeoDataFrame
        Returns a GeoDataFrame representing all towns in the region.

    __getitem__(arg)
        Overloads the subscript operator to access a town or accessibility data based on the argument type.
    
    from_pickle(file_path: str) -> Region
        Loads a `Region` object from a .pickle file.
    
    to_pickle(file_path: str)
        Saves the `Region` object to a .pickle file.
    """

    def __init__(
            self, 
            region : gpd.GeoDataFrame, 
            districts : gpd.GeoDataFrame, 
            settlements : gpd.GeoDataFrame, 
            towns : gpd.GeoDataFrame, 
            accessibility_matrix : pd.DataFrame, 
            territories : gpd.GeoDataFrame | None = None
        ):
        """
        Initializes the Region object with GeoDataFrames for region, districts, settlements, and towns. 
        Optionally includes territories and an accessibility matrix to model transportation between towns.

        Parameters
        ----------
        region : gpd.GeoDataFrame
            GeoDataFrame representing the boundaries of the region.
        districts : gpd.GeoDataFrame
            GeoDataFrame containing information about districts.
        settlements : gpd.GeoDataFrame
            GeoDataFrame containing information about settlements.
        towns : gpd.GeoDataFrame
            GeoDataFrame containing information about towns.
        accessibility_matrix : pd.DataFrame
            DataFrame containing accessibility data between towns.
        territories : gpd.GeoDataFrame, optional
            GeoDataFrame containing information about territories (default is None).

        Raises
        ------
        AssertionError
            If the CRS or indices between the towns and the accessibility matrix do not match.
        """
        region = self.validate_region(region)
        districts = self.validate_districts(districts)
        settlements = self.validate_settlements(settlements)
        towns = self.validate_towns(towns)
        accessibility_matrix = self.validate_accessibility_matrix(accessibility_matrix)

        assert (accessibility_matrix.index == towns.index).all(), "Accessibility matrix indices and towns indices don't match"
        assert region.crs == districts.crs == settlements.crs == towns.crs, 'CRS should match everywhere'

        self.crs = towns.crs
        self.region = region
        self.districts = districts
        self.settlements = settlements
        self._towns = Town.from_gdf(towns)
        self.accessibility_matrix = accessibility_matrix
        if territories is None:
            self._territories = {}
        else:
            assert territories.crs == towns.crs, 'Territories CRS should match towns CRS'
            self._territories = Territory.from_gdf(territories)


    @staticmethod
    def validate_districts(gdf : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Validates the districts GeoDataFrame to ensure it has the correct structure and data types.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            The GeoDataFrame containing district data.

        Returns
        -------
        gpd.GeoDataFrame
            Validated GeoDataFrame with 'geometry' and 'name' columns.

        Raises
        ------
        AssertionError
            If the GeoDataFrame is not of the expected format or types.
        """
        assert isinstance(gdf, gpd.GeoDataFrame), 'Districts should be instance of gpd.GeoDataFrame'
        assert gdf.geom_type.isin(['Polygon', 'MultiPolygon']).all(), 'District geometry should be Polygon or MultiPolygon'
        assert pd.api.types.is_string_dtype(gdf['name']), 'District name should be str'
        return gdf[['geometry', 'name']]

    @staticmethod
    def validate_settlements(gdf : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Validates the settlements GeoDataFrame to ensure it has the correct structure and data types.
        """
        assert isinstance(gdf, gpd.GeoDataFrame), 'Settlements should be instance of gpd.GeoDataFrame'
        assert gdf.geom_type.isin(['Polygon', 'MultiPolygon']).all(), 'Settlement geometry should be Polygon or MultiPolygon'
        assert pd.api.types.is_string_dtype(gdf['name']), 'Settlement name should be str'
        return gdf[['geometry', 'name']]
    
    @staticmethod
    def validate_towns(gdf : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Validates the towns GeoDataFrame.
        """
        assert isinstance(gdf, gpd.GeoDataFrame), 'Towns should be instance of gpd.GeoDataFrame'
        return gdf

    @staticmethod
    def validate_region(gdf : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Validates the region GeoDataFrame to ensure it has the correct structure and data types.
        """
        assert isinstance(gdf, gpd.GeoDataFrame), 'Region should be instance of gpd.GeoDataFrame'
        assert gdf.geom_type.isin(['Polygon', 'MultiPolygon']).all(), 'District geometry should be Polygon or MultiPolygon'
        assert pd.api.types.is_string_dtype(gdf['name']), 'District name should be str'
        return gdf[['geometry', 'name']]
    
    @staticmethod
    def validate_accessibility_matrix(df : pd.DataFrame) -> pd.DataFrame:
        """
        Validates the accessibility matrix, ensuring it has non-negative float values 
        and matching row and column indices.
        """
        assert pd.api.types.is_float_dtype(df.values), 'Accessibility matrix values should be float'
        assert (df.values>=0).all(), 'Accessibility matrix values should be greater or equal 0'
        assert (df.index == df.columns).all(), "Accessibility matrix indices and columns don't match"
        return df

    def plot(self, figsize=(15, 15)):
        """
        Plots the region, including districts, settlements, towns, and optionally territories, on a map.
        
        Parameters
        ----------
        figsize : tuple, optional
            The size of the plot (default is (15, 15)).
        """
        sett_to_dist = self.settlements.copy()
        sett_to_dist.geometry = sett_to_dist.representative_point()
        sett_to_dist = sett_to_dist.sjoin(self.districts).rename(columns={
            'name_right': 'district_name',
            'index_right': 'district_id',
            'name_left': 'settlement_name'
        })
        sett_to_dist.geometry = self.settlements.geometry

        fig, ax = plt.subplots(figsize=figsize)

        # Plot settlements
        sett_to_dist.plot(ax=ax, facecolor='none', edgecolor=SETTLEMENTS_PLOT_COLOR, linewidth=0.7, label='Settlements')

        # Plot towns
        towns_plot = self.get_towns_gdf().plot(ax=ax, markersize=10, color=TOWNS_PLOT_COLOR, marker='o', alpha=0.6, label='Towns')

        # Plot districts
        districts_plot = self.districts.plot(ax=ax, facecolor='none', edgecolor=DISTRICTS_PLOT_COLOR, linewidth=1, label='Districts')
        
        # Plot territories
        territories_gdf = self.get_territories_gdf()
        if territories_gdf is not None:
            territories_gdf.geometry = territories_gdf.representative_point()
            territories_plot = territories_gdf.plot(ax=ax, marker="H", markersize=600, color=TERRITORIES_PLOT_COLOR, alpha=0.9, label='Territories')

        # Additional settings for better visualization
        ax.set_axis_off()

        # Create legend with smaller markers
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles, labels, loc='upper right', scatterpoints=1, markerscale=0.5, handletextpad=1.5)

        plt.title("Region model", fontsize=16)
        plt.tight_layout()
        plt.show()

    
    def get_territory(self, territory_id : int):
        """
        Retrieves a specific territory by its ID.

        Parameters
        ----------
        territory_id : int
            The ID of the territory.

        Returns
        -------
        Territory
            The requested territory.

        Raises
        ------
        KeyError
            If the territory with the given ID is not found.
        """
        if not territory_id in self._territories:
            raise KeyError(f"Can't find territory with such id: {territory_id}")
        return self._territories[territory_id]

    @property
    def towns(self) -> list[Town]:
        """
        Returns a list of all towns in the region.

        Returns
        -------
        list[Town]
            List of Town objects.
        """
        return self._towns.values()
    
    @property
    def territories(self) -> list[Territory]:
        """
        Returns a list of all territories in the region.

        Returns
        -------
        list[Territory]
            List of Territory objects.
        """
        return self._territories.values()

    @property
    def geometry(self) -> shapely.Polygon | shapely.MultiPolygon:
        """
        Returns the unified geometry of all districts in the region, transformed to the WGS84 CRS (EPSG:4326).

        Returns
        -------
        shapely.Polygon or shapely.MultiPolygon
            The unified geometry of the region.
        """
        return self.districts.to_crs(4326).unary_union

    def get_territories_gdf(self) -> gpd.GeoDataFrame:
        """
        Returns a GeoDataFrame representing all territories in the region.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with territory data.
        """
        data = [territory.to_dict() for territory in self.territories]
        return gpd.GeoDataFrame(data, crs=self.crs).set_index('id', drop=True)

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
        gdf = gdf.sjoin(
            self.settlements[['geometry', 'name']].rename(columns={'name': 'settlement_name'}), 
            how='left',
            predicate='within',
            lsuffix='town',
            rsuffix='settlement'
        )
        gdf = gdf.sjoin(
            self.districts[['geometry', 'name']].rename(columns={'name':'district_name'}),
            how='left',
            predicate='within',
            lsuffix='town',
            rsuffix='district'
        )
        return gdf.drop(columns=['index_settlement', 'index_district']).fillna(0)

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
        state = None
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