import bw2calc as bc
import bw2data
import bw_processing as bwp
import xarray as xr

from .lcia import build_characterization_matrix_for_sankey
from .utils import get_unit_conversion_factors, load_units_conversion


class Sankey:
    """
    A class to represent a Sankey diagram.
    """

    def __init__(
        self,
        method: str,
        model: str,
        scenario: str,
        region: str,
        year: int,
        variable: str,
        dp: bwp,
        biosphere_dict: dict,
        activity_dict: dict,
        mapping: dict,
        scenario_data: xr.DataArray,
        cutoff: float = 1e-3,
    ):
        """
        Initialize the Sankey object.
        """

        self.method = method
        self.model = model
        self.scenario = scenario
        self.region = region
        self.year = year
        self.variable = variable
        self.dp = dp
        self.biosphere_dict = biosphere_dict
        self.activity_dict = activity_dict
        self.cutoff = cutoff
        self.scenario_data = scenario_data

        c_data, c_indices, c_unit = build_characterization_matrix_for_sankey(
            method=self.method, biosphere_dict=self.biosphere_dict
        )

        # fetch dataset name of variables
        dataset_name = tuple(list(mapping[self.variable]["dataset"][0].values()))
        # add region to tuple
        dataset_name = dataset_name + (self.region,)
        # getch index of dataset
        dataset_index = self.activity_dict[dataset_name]
        self.dataset_index = dataset_index
        # fetch demand
        demand = self.scenario_data.sel(
            model=self.model,
            pathway=self.scenario,
            region=self.region,
            year=self.year,
            variables=self.variable,
        ).values
        demand_unit = self.scenario_data.attrs["units"][self.variable]
        demand = float(demand) * float(
            get_unit_conversion_factors(
                demand_unit, dataset_name[2], load_units_conversion()
            )
        )

        self.lcia_unit = c_unit

        self.dp.add_persistent_vector(
            matrix="characterization_matrix",
            indices_array=c_indices,
            data_array=c_data,
        )

        self.lca = bc.LCA(
            demand={dataset_index: demand},
            data_objs=[
                self.dp,
            ],
        )
