import pyproj
import shapely
import geopandas as gpd
from pydantic import BaseModel, InstanceOf


class Territory(BaseModel):
    """
    A class representing a geographic territory with an ID, name, and geometry.

    Attributes
    ----------
    id : int
        Unique identifier for the territory.
    name : str
        Name of the territory.
    geometry : shapely.Polygon
        A Shapely Polygon object representing the territory's geometry.

    Methods
    -------
    to_dict() -> dict
        Converts the Territory object to a dictionary format.

    from_gdf(cls, gdf: gpd.GeoDataFrame) -> list
        Class method to create a list of Territory instances from a GeoDataFrame.
    """

    id: int
    name: str
    geometry: InstanceOf[shapely.Polygon]

    def to_dict(self) -> dict:
        """
        Converts the Territory instance to a dictionary.

        Returns
        -------
        dict
            A dictionary containing the 'id', 'name', and 'geometry' of the territory.
        """
        return {
            'id': self.id,
            'name': self.name,
            'geometry': self.geometry
        }

    @classmethod
    def from_gdf(cls, gdf: gpd.GeoDataFrame) -> list:
        """
        Creates a dictionary of Territory objects from a GeoDataFrame.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing data to initialize Territory instances. 
            The GeoDataFrame should have 'id', 'name', and 'geometry' columns.

        Returns
        -------
        dict
            A dictionary where the keys are indices of the GeoDataFrame and the values are Territory instances.
        """
        return {i: cls(id=i, **gdf.loc[i].to_dict()) for i in gdf.index}
