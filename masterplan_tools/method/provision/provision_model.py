"""
This module is aimed to provide all necessary tools to get an estimation of provision
of the selected service
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from masterplan_tools.models import CityModel

tqdm.pandas()


class ProvisionModel:
    """
    This class represents a model for calculating the provision of a specified service in a city.

    Attributes:
        blocks (gpd.GeoDataFrame): A GeoDataFrame containing information about the blocks in the city.
        service_name (str): The name of the service for which the provision is being calculated.
        standard (int): The standard value for the specified service, taken from the `standard_dict` attribute.
        g (int): The value of the `g` attribute.
    """

    standard_dict = {
        "kindergartens": 61,
        "schools": 120,
        "universities": 13,
        "hospitals": 9,
        "policlinics": 27,
        "theaters": 5,
        "cinemas": 10,
        "cafes": 72,
        "bakeries": 72,
        "fastfoods": 72,
        "music_school": 8,
        "sportgrounds": 15,
        "swimming_pools": 50,
        "conveniences": 90,
        "recreational_areas": 6000,
        "pharmacies": 50,
        "playgrounds": 550,
        "supermarkets": 992,
    }

    services_accessibility_dict = {
        "kindergartens": 10,
        "schools": 15,
        "universities": 60,
        "hospitals": 60,
        "policlinics": 15,
        "theaters": 60,
        "cinemas": 60,
        "cafes": 30,
        "bakeries": 30,
        "fastfoods": 30,
        "music_school": 30,
        "sportgrounds": 10,
        "swimming_pools": 30,
        "conveniences": 8,
        "recreational_areas": 15,
        "pharmacies": 10,
        "playgrounds": 5,
        "supermarkets": 30,
    }

    def __init__(
        self,
        city_model: CityModel,
        service_name: str = "schools",
    ):
        self.blocks = city_model.blocks.to_gdf().copy()
        self.service_name = service_name
        self.standard = self.standard_dict[self.service_name]
        self.accessibility = self.services_accessibility_dict[self.service_name]
        self.graph = city_model.services_graph.copy()
        self.blocks_aggregated = city_model.blocks.to_gdf().copy()


    def get_provision(self, overflow: bool = False):  # pylint: disable=too-many-branches,too-many-statements
        """
        This function calculates the provision of a specified service in a city.
        The provision is calculated based on the data in the `g` attribute of the object.

        Returns:
            nx.Graph: A networkx graph representing the city's road network with updated data
            about the provision of the specified service.
        """

        graph = self.graph.copy()
        standard = self.standard
        accessibility = self.accessibility

        # if not overflow:
        #     for u, v, data in list(graph.edges(data=True)):  # pylint: disable=invalid-name
        #         if data["weight"] > accessibility:
        #             graph.remove_edge(u, v)

        #     for node in list(graph.nodes):
        #         if graph.degree(node) == 0 and graph.nodes[node][f"is_{self.service_name}_service"] != 1:
        #             graph.remove_node(node)

        for node in graph.nodes:
            if graph.nodes[node]["is_living"]:
                graph.nodes[node][f"population_unprov_{self.service_name}"] =(
                    graph.nodes[node]["population"]/ 1000 * standard)
                graph.nodes[node][f"demand_{self.service_name}"] =  graph.nodes[node][f"population_unprov_{self.service_name}"]
                graph.nodes[node][f"count_{self.service_name}"] = graph.nodes[node][f"demand_{self.service_name}"]
                graph.nodes[node][f"small_prov_{self.service_name}"] = 0


        total_load = 0
        total_capacity = 0


        for node in graph.nodes:
            if graph.nodes[node][f"is_{self.service_name}_service"] == 1:
                total_capacity += graph.nodes[node][f"{self.service_name}_capacity"]

            if graph.nodes[node]["is_living"] and graph.nodes[node]["population"] > 0:
                total_load += graph.nodes[node][f"count_{self.service_name}"]
                               
        print(f'total load = {total_load}')
        print(f'total capacity = {total_capacity}')
        
        count = 0
        load = 1

        
        while total_load > 0 or total_capacity > 0:
            prev_total_load = total_load
            prev_total_capacity = total_capacity

            for node in graph.nodes:
                # print('break')
                if (graph.nodes[node]["is_living"]
                    and graph.nodes[node][f"count_{self.service_name}"] > 0
                ):
                    if (graph.nodes[node][f"is_{self.service_name}_service"] == 1
                        and graph.nodes[node][f"{self.service_name}_capacity"] >= 1
                    ):
                        if (graph.nodes[node][f"provision_{self.service_name}"] < 100
                            and total_capacity >0
                            and total_load > 0 
                        ):
                            graph.nodes[node][f"{self.service_name}_capacity"] -=load
                            total_capacity -= load
                            graph.nodes[node][f"population_prov_{self.service_name}"] += load
                            total_load -= load
                            graph.nodes[node][f"count_{self.service_name}"] -= load
                            graph.nodes[node][f"population_unprov_{self.service_name}"] -= load
                            graph.nodes[node][f"provision_{self.service_name}"] = (
                                graph.nodes[node][f"population_prov_{self.service_name}"] * 100
                                / graph.nodes[node][f"demand_{self.service_name}"]
                            )
                            
                    else:
                        neighbors = list(graph.neighbors(node))
                        neighbors_sorted = sorted(neighbors, key=lambda neighbor: graph[node][neighbor]["weight"])
                        for neighbor in neighbors_sorted:
                            if (graph.nodes[neighbor][f"is_{self.service_name}_service"] == 1
                            and graph.nodes[neighbor][f"{self.service_name}_capacity"] >= 1
                            and total_capacity >0
                            and total_load > 0 
                            ):
                                if graph[node][neighbor]["weight"] <= accessibility:
                                    graph.nodes[node][f"count_{self.service_name}"] -= load
                                    graph.nodes[neighbor][f"{self.service_name}_capacity"] -= load
                                    total_capacity -= load
                                    graph.nodes[node][f"population_prov_{self.service_name}"] += load
                                    total_load -= load
                                    graph.nodes[node][f"population_unprov_{self.service_name}"] -= load
                                    graph.nodes[node][f"provision_{self.service_name}"] = (
                                        graph.nodes[node][f"population_prov_{self.service_name}"] * 100
                                        / graph.nodes[node][f"demand_{self.service_name}"]
                                    )
                                    break
                                else: 
                                    graph.nodes[neighbor][f"{self.service_name}_capacity"] -= load
                                    total_capacity -= load
                                    graph.nodes[node][f"count_{self.service_name}"] -= load
                                    total_load -= load
                                    graph.nodes[node][f"small_prov_{self.service_name}"] +=1
                                    break


            # count+=1
            # if total_capacity < 0:
            #     print('c4')
            #     print(total_load)
            #     print(total_capacity)
            #     total_capacity = prev_total_capacity
            #     print(count)
            #     break
            # if total_load < 0:
            #     print('l4')
            #     print(total_load)
            #     print(total_capacity)
            #     total_load = prev_total_load
            #     print(count)
            #     break


            if prev_total_load == total_load or prev_total_capacity == total_capacity:
                print('fin')
                print(f' load = {total_load}')
                print(f' cap = {total_capacity}')
                print(count)
                break
        
        self.graph = graph



    def set_blocks_attributes(self) -> pd.DataFrame:
        """
        This function returns a copy of the `blocks` attribute of the object with updated values for the service
        specified by the `service_name` attribute. The values are updated based on the data in the `graph` attribute
        of the object.

        Returns:
            DataFrame: A copy of the `blocks` attribute with updated values for the specified service.
        """
        standard = self.standard
        graph = self.graph.copy()
        blocks = self.blocks.copy()

        new_columns = [f"provision_{self.service_name}", f"population_prov_{self.service_name}",
                    f"population_unprov_{self.service_name}", f"small_prov_{self.service_name}"]

        for col in new_columns:
            blocks[col] = 0

        blocks["population"] = 0

        # Process nodes in the graph
        for node in graph:
            if graph.nodes[node]["is_living"]:
                indx = blocks.index.get_loc(node)
                node_data = graph.nodes[node]

                for col in new_columns:
                    if col in node_data:
                        blocks.at[indx, col] = node_data[col]

                blocks.at[indx, "population"] = node_data["population"]

        if self.service_name == "recreational_areas":
            blocks[f"population_unprov_{self.service_name}"] = blocks[f"population_unprov_{self.service_name}"]

        # for i in range(len(blocks)):
        #     if blocks.loc[i, "population"] == 0:
        #         blocks.loc[i, "population"] = blocks.loc[i, "current_population"]
        #         blocks.loc[i, f"population_unprov_{self.service_name}"] = blocks.loc[i, "current_population"] /1000* standard


        int_columns = [f"population_prov_{self.service_name}", f"population_unprov_{self.service_name}",
                    f"small_prov_{self.service_name}",
                    f"provision_{self.service_name}", "population"]

        blocks[int_columns] = blocks[int_columns].astype(int)

        blocks[f"provision_{self.service_name}"] = np.minimum(blocks[f"provision_{self.service_name}"], 100)

        # blocks = blocks.drop(columns=[f"id_{self.service_name}"])

       
        return blocks

    def run(self, overflow: bool = False):
        """
        This function runs the model to calculate the provision of a specified service in a city.
        The function calls the `get_stats`, `get_provision`, and `get_geo` methods of the object.

        Returns:
            DataFrame: A DataFrame containing information about the provision of the specified service in the city.
        """

        
        self.get_provision(overflow=overflow)
        return self.set_blocks_attributes()
