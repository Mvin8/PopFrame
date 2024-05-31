
import pyproj
import shapely
import geopandas as gpd

from pydantic import BaseModel, InstanceOf

class Territory(BaseModel):
  id : int
  name : str
  geometry : InstanceOf[shapely.Polygon]

  def to_dict(self):
    return {
      'id': self.id,
      'name': self.name,
      'geometry': self.geometry
    }

  @classmethod
  def from_gdf(cls, gdf : gpd.GeoDataFrame) -> list:
    return {i:cls(id=i, **gdf.loc[i].to_dict()) for i in gdf.index}