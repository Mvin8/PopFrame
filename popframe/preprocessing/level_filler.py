import geopandas as gpd
from pydantic import BaseModel, Field, field_validator
from shapely.geometry import Point
from typing import ClassVar
from ..models.geodataframe import GeoDataFrame, BaseRow


class TownRow(BaseRow):
    geometry: Point
    name: str
    rural_settlement: str
    level: str = Field(default="Нет уровня")


class LevelFiller(BaseModel):
    towns: GeoDataFrame[TownRow]
    administrative_centers: ClassVar[list[str]] = [
        "Ломоносов",
        "Тосно",
        "Тихвин",
        "Сосновый бор",
        "Сланцы",
        "Приозерск",
        "Подпорожье",
        "Луга",
        "Лодейное Поле",
        "Кировск",
        "Кириши",
        "Кингисепп",
        "Гатчина",
        "Выборг",
        "Всеволожск",
        "Волхов",
        "Волосово",
        "Бокситогорск"
    ]

    @staticmethod
    def _assign_level(row) -> str:
        rural_settlement = row['rural_settlement'].strip()
        name = row['name']
        
        if rural_settlement in ['деревня', 'село', 'хутор', 'местечко', 'кордон', 'деревня']:
            return 'Пятый уровень'
        elif rural_settlement in ['город', 'поселок', 'городской поселок', 'поселок при железнодорожной станции']:
            return 'Четвертый уровень'
        elif 'административный центр' in rural_settlement:
            if name in LevelFiller.administrative_centers:
                return 'Второй уровень'
            return 'Третий уровень'
        return 'Нет уровня'

    @field_validator("towns", mode="before")
    @classmethod
    def validate_towns(cls, gdf):
        gdf.rename(columns={'rural settlement': 'rural_settlement'}, inplace=True)
        gdf["level"] = gdf.apply(cls._assign_level, axis=1)
        return GeoDataFrame[TownRow](gdf)

    def fill_levels(self) -> gpd.GeoDataFrame:
        towns = self.towns.copy()
        towns["level"] = towns.apply(self._assign_level, axis=1)
        return towns
