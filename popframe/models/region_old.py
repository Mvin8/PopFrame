from functools import singledispatchmethod
from pydantic import BaseModel, InstanceOf
import dill as pickle
import shapely
import geopandas as gpd
import pandas as pd
from .geodataframe import GeoDataFrame, BaseRow


DISTRICTS_PLOT_COLOR = '#e40e37'
SETTLEMENTS_PLOT_COLOR = '#ddd'
TOWNS_PLOT_COLOR = '#777'
SERVICES_PLOT_COLOR = '#aaa'
TERRITORIES_PLOT_COLOR = '#29392C'

class DistrictsRow(BaseRow):
    name : str
    geometry : shapely.Polygon | shapely.MultiPolygon

class SettlementsRow(BaseRow):
    name : str
    geometry : shapely.Polygon | shapely.MultiPolygon

class TownRow(BaseRow):
    name : str
    level: str
    geometry : shapely.Point
    population : int

class TerRow(BaseRow):
    id : int
    name : str
    geometry : shapely.Polygon | shapely.MultiPolygon

class Town(BaseModel):
    id : int
    name : str
    population : int
    level: str
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
    def from_gdf(cls, gdf):
        res = {}
        for i in gdf.index:
            res[i] = cls(id=i, **gdf.loc[i].to_dict())
        return res 
    

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
    
class Region():

    def __init__(self, districts : GeoDataFrame[DistrictsRow] | gpd.GeoDataFrame, settlements : GeoDataFrame[SettlementsRow] | gpd.GeoDataFrame, towns : GeoDataFrame[TownRow] | gpd.GeoDataFrame, adjacency_matrix : pd.DataFrame, territories : GeoDataFrame[TerRow] | gpd.GeoDataFrame,):
        if not isinstance(districts, GeoDataFrame[DistrictsRow]):
            districts = GeoDataFrame[DistrictsRow](districts)
        if not isinstance(settlements, GeoDataFrame[SettlementsRow]):
            settlements = GeoDataFrame[SettlementsRow](settlements)
        if not isinstance(towns, GeoDataFrame[TownRow]):
            towns = GeoDataFrame[TownRow](towns)
        if not isinstance(territories, GeoDataFrame[TerRow]):
            territories = GeoDataFrame[DistrictsRow](territories)
        assert (adjacency_matrix.index == adjacency_matrix.columns).all(), "Adjacency matrix indices and columns don't match"
        assert (adjacency_matrix.index == towns.index).all(), "Adjacency matrix indices and towns indices don't match"
        assert(districts.crs == settlements.crs and settlements.crs == towns.crs and territories.crs == settlements.crs), 'CRS should march everywhere'
        self.crs = towns.crs
        self.districts = districts
        self.territories = territories
        self.settlements = settlements
        self.adjacency_matrix = adjacency_matrix
        self._towns = Town.from_gdf(towns)
        self._territory = Territory.from_gdf(territories)
        

    def get_towns_gdf(self) -> gpd.GeoDataFrame:
        data = [town.to_dict() for town in self.towns]
        gdf = gpd.GeoDataFrame(data, crs=self.crs).rename(columns={'name': 'town_name'}).set_index('id', drop=True)
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
    

    def get_territories_gdf(self) -> gpd.GeoDataFrame:
        data = [territory.territories_to_dict() for territory in self.territories]
        return gpd.GeoDataFrame(data, crs=self.crs).set_index('id', drop=True)
    

    @singledispatchmethod
    def __getitem__(self, arg):
        raise NotImplementedError(f"Can't access object with such argument type {type(arg)}")

    # Make city_model subscriptable, to access block via ID like city_model[123]
    @__getitem__.register(int)
    def _(self, town_id):
        if not town_id in self._towns:
            raise KeyError(f"Can't find town with such id: {town_id}")
        return self._towns[town_id]

    # Make city_model subscriptable, to access service type via name like city_model['schools']

    @__getitem__.register(tuple)
    def _(self, towns):
        (town_a, town_b) = towns
        if isinstance(town_a, Town):
            town_a = town_a.id
        if isinstance(town_b, Town):
            town_b = town_b.id
        return self.adjacency_matrix.loc[town_a, town_b]

    @property
    def towns(self):
        return self._towns.values()
    
    @property
    def territory(self):
        return self._territory.values()
    
    def plot(self, figsize=(15,15)):
        sett_to_dist = self.settlements.copy()
        sett_to_dist.geometry = sett_to_dist.representative_point()
        sett_to_dist = sett_to_dist.sjoin(self.districts).rename(columns={
        'name_right':'district_name',
        'index_right': 'district_id',
        'name_left': 'settlement_name'
        })
        sett_to_dist.geometry = self.settlements.geometry
        ax = sett_to_dist.plot(facecolor='none', edgecolor=SETTLEMENTS_PLOT_COLOR, figsize=figsize)
        self.get_towns_gdf().plot(ax=ax, markersize=2, color=TOWNS_PLOT_COLOR)
        self.districts.plot(ax=ax, facecolor='none', edgecolor=DISTRICTS_PLOT_COLOR)
        territories_gdf = self.get_territories_gdf()
        territories_gdf.geometry = territories_gdf.representative_point()
        territories_gdf.plot(ax=ax, marker="*", markersize=60, color=TERRITORIES_PLOT_COLOR)
        ax.set_axis_off()

    @staticmethod
    def from_pickle(file_path: str):
        """Load region model from a .pickle file"""
        state = None
        with open(file_path, "rb") as f:
            state = pickle.load(f)
        return state

    def to_pickle(self, file_path: str):
        """Save region model to a .pickle file"""
        with open(file_path, "wb") as f:
            pickle.dump(self, f)