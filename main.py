from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from popframe.method.territory_evaluation import TerritoryEvaluation
from popframe.method.urbanisation_level import UrbanisationLevel
from popframe.models.region import Region
from popframe.method.popuation_frame import PopFrame
import geopandas as gpd
import json

app = FastAPI()

class GeoJSONRequest(BaseModel):
    type: str
    name: str
    crs: dict
    features: list

class CriteriaRequest(BaseModel):
    population: int
    transport: int
    ecology: int
    social_objects: int
    engineering_infrastructure: int

class EvaluateTerritoryLocationResult(BaseModel):
    territory: str
    score: int
    interpretation: str
    closest_settlement: str
    closest_settlement1: Optional[str]
    closest_settlement2: Optional[str]

class PopulationCriterionResult(BaseModel):
    project: str
    average_population_density: float
    average_population_growth: float
    score: int

class CalculatePotentialResult(BaseModel):
    category: str
    scores: List[int]

class BuildNetworkResult(BaseModel):
    geojson: Dict[str, Any]

region_model = Region.from_pickle('C:\Code\PopFrame\examples\data\model_data/region_noter.pickle')
# gdf = gpd.read_file('examples\data\model_data\growth_mo.geojson')

if not isinstance(region_model, Region):
    raise Exception("Invalid region model")

@app.get("/", response_model=Dict[str, str])
def read_root():
    return {"message": "Welcome to PopFrame Service"}

@app.post("/evaluate_territory_location", response_model=List[EvaluateTerritoryLocationResult])
async def evaluate_territory_location_endpoint(request: GeoJSONRequest):
    try:
        evaluation = TerritoryEvaluation(region=region_model)
        result = evaluation.evaluate_territory_location(territories=request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/population_criterion", response_model=List[PopulationCriterionResult])
async def population_criterion_endpoint(request: GeoJSONRequest):
    try:
        evaluation = TerritoryEvaluation(region=region_model)
        result = evaluation.population_criterion(gdf, territories=request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/calculate_potential", response_model=List[CalculatePotentialResult])
async def calculate_potential_endpoint(request: CriteriaRequest):
    try:
        territories_criteria = {
            "Население": request.population,
            "Транспорт": request.transport,
            "Экология": request.ecology,
            "Соц-об": request.social_objects,
            "Инж инф": request.engineering_infrastructure,
        }

        evaluation = TerritoryEvaluation(region=region_model)
        result = evaluation.calculate_potential(territories_criteria)
        
        formatted_result = [
            {"category": category, "scores": scores}
            for category, scores in result
        ]
        
        return formatted_result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/build_network_frame", response_model=Dict[str, Any])
async def build_network_endpoint():
    try:
        frame_method = PopFrame(region=region_model)
        G = frame_method.build_network_frame()
        gdf_frame = frame_method.save_graph_to_geojson(G, None)
        
        # Возвращаем GeoDataFrame как GeoJSON
        return json.loads(gdf_frame.to_json())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")
    

@app.post("/build_square_frame", response_model=Dict[str, Any])
async def build_network_endpoint():
    try:
        frame_method = PopFrame(region=region_model)
        gdf_frame = frame_method.build_square_frame(output_type='gdf')
        # Возвращаем GeoDataFrame как GeoJSON
        return json.loads(gdf_frame.to_json())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")
    
@app.post("/get_landuse_data", response_model=Dict[str, Any])
async def get_landuse_data_endpoint(request: GeoJSONRequest):
    try:
        urbanisation = UrbanisationLevel(region=region_model)
        landuse_data = urbanisation.get_landuse_data(territories=request.dict())
        
        # Convert the resulting GeoDataFrame to GeoJSON format for response
        return json.loads(landuse_data.to_json())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


