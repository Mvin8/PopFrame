from pydantic import BaseModel, InstanceOf, field_validator, Field
import pyproj
import shapely
import geopandas as gpd
import pandas as pd
     
class Town(BaseModel):
    id : int
    name : str
    level: str
    population : int = Field(gt=0)
    geometry : InstanceOf[shapely.Point]

    def to_dict(self):
        res = {
            'id': self.id,
            'name': self.name,
            'population': self.population,
            'level' : self.level,
            'geometry': self.geometry
        }
        return res
    
    @classmethod
    def from_gdf(cls, gdf) -> dict:
        return {i:cls(id=i, **gdf.loc[i].to_dict()) for i in gdf.index}
