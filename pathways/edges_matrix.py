"""
This module defines runs Edges, to produce a (regionalized) characterization matrix.
"""

import json
import logging
import re
import uuid
from typing import Dict, Optional

import bw2calc
import numpy as np
import pandas as pd
import sparse as spnd
from bw_processing import Datapackage
from edges import EdgeLCIA
from edges import setup_package_logging
from edges.matrix_builders import build_technosphere_edges_matrix
from scipy.sparse import csr_matrix, issparse, vstack

from .filesystem_constants import DATA_DIR, DIR_CACHED_DB

setup_package_logging(level=logging.DEBUG)


def fetch_topology(model: str) -> Optional[Dict]:
    """
    Find the JSON file containing the topologies of the provided model.
    """
    topology_path = DATA_DIR / "topologies" / f"{model.lower()}-topology.json"
    if topology_path.exists():
        # load json
        return json.loads(topology_path.read_text())

    raise FileNotFoundError(
        f"Geographical definition file for the model '{model.upper()}' not found."
    )


def _edge_sets_for_lookup(lca: EdgeLCIA):
    # Start empty; we'll OR them depending on which edge family you use
    restrict_supplier_positions_bio: set[int] = set()
    restrict_supplier_positions_tech: set[int] = set()
    restrict_consumer_positions: set[int] = set()

    if getattr(lca, "biosphere_edges", None):
        # biosphere_edges: (bio_row, tech_col)
        bio_rows = {r for (r, _c) in lca.biosphere_edges}
        tech_cols = {c for (_r, c) in lca.biosphere_edges}
        restrict_supplier_positions_bio |= bio_rows
        restrict_consumer_positions |= tech_cols

    if getattr(lca, "technosphere_edges", None):
        # technosphere_edges: (tech_row_supplier, tech_col_consumer)
        tech_rows = {r for (r, _c) in lca.technosphere_edges}
        tech_cols = {c for (_r, c) in lca.technosphere_edges}
        restrict_supplier_positions_tech |= tech_rows
        restrict_consumer_positions |= tech_cols

    return (
        restrict_supplier_positions_bio,
        restrict_supplier_positions_tech,
        restrict_consumer_positions,
    )


def _build_position_to_technosphere_lookup(
    technosphere_index: dict[int, dict],
) -> dict[int, dict]:
    """
    technosphere_index: maps position -> activity metadata (name, location, classifications, etc.)
    Return minimal fields Edges uses to enrich consumer/supplier info.
    """
    out = {}
    for pos, meta in technosphere_index.items():
        out[pos] = {
            "location": meta.get("location"),
            "classifications": meta.get("classifications"),
            "name": meta.get("name"),
            "reference product": meta.get("reference product"),
            "unit": meta.get("unit"),
        }
    return out


def _ensure_minimal_flows(
    lca: EdgeLCIA,
    biosphere_index: dict[int, dict],
    technosphere_index: dict[int, dict],
):
    if not getattr(lca, "reversed_activity", None):
        lca.reversed_activity = {v: k for k, v in lca.lca.dicts.activity.items()}

    if not getattr(lca, "reversed_biosphere", None):
        lca.reversed_biosphere = {v: k for k, v in lca.lca.dicts.biosphere.items()}

    if not getattr(lca, "biosphere_flows", None):
        lca.biosphere_flows = [
            {
                "name": f.get("name"),
                "categories": list(f.get("categories")),
                "unit": f.get("unit"),
                "location": f.get("location"),  # usually None for biosphere flows
                "classifications": f.get("classifications"),  # optional
                "position": lca.lca.dicts.biosphere[pos],
            }
            for pos, f in biosphere_index.items()
            if pos in lca.lca.dicts.biosphere
        ]

    if not getattr(lca, "technosphere_flows", None):
        lca.technosphere_flows = [
            {
                "name": a.get("name"),
                "reference product": a.get("reference product"),
                "unit": a.get("unit"),
                "location": a.get("location"),
                "classifications": a.get("classifications"),
                "position": lca.lca.dicts.activity.reversed[pos],
            }
            for pos, a in technosphere_index.items()
            if pos in lca.lca.dicts.activity
        ]


def _as_row_csr(mat):
    """Ensure the characterization is a 2D CSR row matrix."""
    if issparse(mat):
        m = mat.tocsr()
        # If it's a column/row vector, normalize to (1, n)
        if m.ndim == 2 and m.shape[0] == 1:
            return m
        if m.ndim == 2 and m.shape[1] == 1:
            return m.T.tocsr()
        return m  # already 2D
    # Dense / 1D -> make (1, n)
    arr = np.atleast_2d(np.asarray(mat))
    if arr.shape[0] != 1 and arr.shape[1] == 1:
        arr = arr.T
    return csr_matrix(arr)


def _edges_method_is_technosphere(lca: EdgeLCIA) -> bool:
    return all(
        cf["supplier"].get("matrix") == "technosphere" for cf in lca.raw_cfs_data
    )


def _coerce_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0].item()
    return value


def _coerce_float(value) -> float:
    value = _coerce_scalar(value)
    if value is None:
        return 0.0
    return float(value)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_").lower()
    return slug or "unknown"


def _build_position_lookup(flows: list[dict] | None) -> dict[int, dict]:
    return {
        flow["position"]: flow
        for flow in (flows or [])
        if flow is not None and "position" in flow
    }


def _build_edges_contributor_table(
    edge_lca: EdgeLCIA,
    inventory: csr_matrix,
    *,
    include_unmatched: bool = False,
) -> pd.DataFrame:
    """Build a contributor table from EDGES mappings without Brightway lookups."""

    is_technosphere = _edges_method_is_technosphere(edge_lca)
    technosphere_lookup = _build_position_lookup(
        getattr(edge_lca, "technosphere_flows", None)
    )
    biosphere_lookup = _build_position_lookup(
        getattr(edge_lca, "biosphere_flows", None)
    )

    def _supplier_metadata(position: int) -> dict:
        lookup = technosphere_lookup if is_technosphere else biosphere_lookup
        return lookup.get(position, {})

    def _consumer_metadata(position: int) -> dict:
        return technosphere_lookup.get(position, {})

    rows: list[dict] = []

    if (
        edge_lca.use_distributions
        and hasattr(edge_lca, "characterization_matrix")
        and getattr(edge_lca, "iterations", 0) > 1
    ):
        cm = edge_lca.characterization_matrix
        for i, j in zip(*cm.sum(axis=2).nonzero()):
            amount = _coerce_float(inventory[i, j])
            if amount == 0:
                continue

            supplier = _supplier_metadata(i)
            consumer = _consumer_metadata(j)
            samples = np.asarray(cm[i, j, :].todense()).reshape(-1).astype(float)
            impact_samples = amount * samples
            cf_p = np.percentile(samples, [5, 25, 50, 75, 95])
            impact_p = np.percentile(impact_samples, [5, 25, 50, 75, 95])

            row = {
                "supplier name": supplier.get("name"),
                "consumer name": consumer.get("name"),
                "consumer reference product": consumer.get("reference product"),
                "consumer location": consumer.get("location"),
                "amount": amount,
                "CF (mean)": samples.mean(),
                "CF (std)": samples.std(),
                "CF (min)": samples.min(),
                "CF (5th)": cf_p[0],
                "CF (25th)": cf_p[1],
                "CF (50th)": cf_p[2],
                "CF (75th)": cf_p[3],
                "CF (95th)": cf_p[4],
                "CF (max)": samples.max(),
                "impact (mean)": impact_samples.mean(),
                "impact (std)": impact_samples.std(),
                "impact (min)": impact_samples.min(),
                "impact (5th)": impact_p[0],
                "impact (25th)": impact_p[1],
                "impact (50th)": impact_p[2],
                "impact (75th)": impact_p[3],
                "impact (95th)": impact_p[4],
                "impact (max)": impact_samples.max(),
            }
            if is_technosphere:
                supplier_product = supplier.get("reference product")
                row["supplier reference product"] = supplier_product
                row["supplier product"] = supplier_product
                row["supplier location"] = supplier.get("location")
            else:
                row["supplier categories"] = supplier.get("categories")
            rows.append(row)
    else:
        for cf in getattr(edge_lca, "scenario_cfs", []):
            for i, j in cf.get("positions", []):
                amount = _coerce_float(inventory[i, j])
                if amount == 0:
                    continue

                supplier = _supplier_metadata(i)
                consumer = _consumer_metadata(j)
                cf_value = _coerce_scalar(cf.get("value"))
                impact = None if cf_value is None else amount * float(cf_value)

                row = {
                    "supplier name": supplier.get("name"),
                    "consumer name": consumer.get("name"),
                    "consumer reference product": consumer.get("reference product"),
                    "consumer location": consumer.get("location"),
                    "amount": amount,
                    "CF": cf_value,
                    "impact": impact,
                }
                if is_technosphere:
                    supplier_product = supplier.get("reference product")
                    row["supplier reference product"] = supplier_product
                    row["supplier product"] = supplier_product
                    row["supplier location"] = supplier.get("location")
                else:
                    row["supplier categories"] = supplier.get("categories")
                rows.append(row)

    if include_unmatched:
        unmatched_edges = (
            getattr(edge_lca, "unprocessed_technosphere_edges", set())
            if is_technosphere
            else getattr(edge_lca, "unprocessed_biosphere_edges", set())
        )
        for i, j in unmatched_edges:
            amount = _coerce_float(inventory[i, j])
            if amount == 0:
                continue

            supplier = _supplier_metadata(i)
            consumer = _consumer_metadata(j)
            row = {
                "supplier name": supplier.get("name"),
                "consumer name": consumer.get("name"),
                "consumer reference product": consumer.get("reference product"),
                "consumer location": consumer.get("location"),
                "amount": amount,
                "CF": None,
                "impact": None,
            }
            if is_technosphere:
                supplier_product = supplier.get("reference product")
                row["supplier reference product"] = supplier_product
                row["supplier product"] = supplier_product
                row["supplier location"] = supplier.get("location")
            else:
                row["supplier categories"] = supplier.get("categories")
            rows.append(row)

    return pd.DataFrame(rows)


def export_edges_contributors(
    *,
    multilca_obj: bw2calc.MultiLCA,
    edges_lcas: dict,
    model: str,
    scenario: str,
    year: int,
    region: str,
    functional_units: dict[str, dict],
    include_unmatched: bool = False,
) -> list[dict]:
    """Export per-functional-unit EDGES contributor tables to cached parquet files."""

    manifest: list[dict] = []

    for method, edge_lca in edges_lcas.items():
        method_label = str(method)
        method_slug = _slugify(method_label)
        method_unit = getattr(edge_lca, "method_metadata", {}).get("unit")
        is_technosphere = _edges_method_is_technosphere(edge_lca)

        for variable, inventory in multilca_obj.inventories.items():
            matrix = inventory.tocsr() if issparse(inventory) else csr_matrix(inventory)
            df = _build_edges_contributor_table(
                edge_lca,
                matrix,
                include_unmatched=include_unmatched,
            )
            if df.empty:
                continue

            fu_meta = functional_units.get(variable, {})
            dataset = fu_meta.get("dataset") or ()
            demand = _coerce_scalar(fu_meta.get("demand"))
            fu_id = _coerce_scalar(fu_meta.get("id"))

            df.insert(0, "method unit", method_unit)
            df.insert(0, "method", method_label)
            df.insert(
                0,
                "functional unit location",
                dataset[3] if len(dataset) > 3 else None,
            )
            df.insert(
                0,
                "functional unit unit",
                dataset[2] if len(dataset) > 2 else None,
            )
            df.insert(
                0,
                "functional unit reference product",
                dataset[1] if len(dataset) > 1 else None,
            )
            df.insert(
                0,
                "functional unit name",
                dataset[0] if len(dataset) > 0 else None,
            )
            df.insert(0, "functional unit id", fu_id)
            df.insert(0, "functional unit demand", demand)
            df.insert(0, "variable", variable)
            df.insert(0, "region", region)
            df.insert(0, "year", year)
            df.insert(0, "scenario", scenario)
            df.insert(0, "model", model)

            filepath = DIR_CACHED_DB / (
                f"edges_contributors_{method_slug}_{_slugify(variable)}_"
                f"{uuid.uuid4().hex}.gzip"
            )
            df.to_parquet(filepath, compression="gzip", index=False)

            manifest.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "year": year,
                    "region": region,
                    "variable": variable,
                    "method": method_label,
                    "method_unit": method_unit,
                    "functional_unit_name": dataset[0] if len(dataset) > 0 else None,
                    "functional_unit_reference_product": (
                        dataset[1] if len(dataset) > 1 else None
                    ),
                    "functional_unit_unit": dataset[2] if len(dataset) > 2 else None,
                    "functional_unit_location": (
                        dataset[3] if len(dataset) > 3 else None
                    ),
                    "functional_unit_id": fu_id,
                    "functional_unit_demand": demand,
                    "rows": int(len(df)),
                    "filepath": str(filepath),
                }
            )

    return manifest


def create_edges_characterization_matrix(
    model: str,
    multilca_obj: bw2calc.MultiLCA,
    methods: list,
    indices: dict[str, dict[int, dict]],
):
    """
    Run Edges for each method and return:
    - a 3D sparse tensor of shape ``(n_methods, m, n)``
    - the updated ``MultiLCA`` object
    - the per-method ``EdgeLCIA`` objects used to build the tensor
    """
    topology = fetch_topology(model)

    planes = []
    edges_lcas = {}

    for method in methods:
        # create fake sparse inventory matrix with same SHAPE & DTYPE
        first_matrix = next(iter(multilca_obj.inventories.values()))
        multilca_obj.inventory = csr_matrix(
            first_matrix.shape, dtype=getattr(first_matrix, "dtype", float)
        )

        lca = EdgeLCIA(
            demand={},
            method=method,
            lca=multilca_obj,
            additional_topologies=topology,
        )

        if all(
            cf["supplier"].get("matrix") == "technosphere" for cf in lca.raw_cfs_data
        ):
            multilca_obj.inventories = {
                k: build_technosphere_edges_matrix(
                    multilca_obj.technosphere_matrix, multilca_obj.supply_arrays[k]
                )
                for k in multilca_obj.supply_arrays
            }

            lca.technosphere_edges = {
                (r, c)
                for mat in multilca_obj.inventories.values()
                for r, c in zip(*mat.nonzero())
            }
            lca.technosphere_flow_matrix = next(iter(multilca_obj.inventories.values()))
        else:
            lca.biosphere_edges = {
                (r, c)
                for mat in multilca_obj.inventories.values()
                for r, c in zip(*mat.nonzero())
            }

        _ensure_minimal_flows(
            lca,
            biosphere_index=indices["biosphere"],
            technosphere_index=indices["technosphere"],
        )
        lca.position_to_technosphere_flows_lookup = (
            _build_position_to_technosphere_lookup(indices["technosphere"])
        )

        rs_bio, rs_tech, rc_cons = _edge_sets_for_lookup(lca)
        lca._preprocess_lookups(
            restrict_supplier_positions_bio=rs_bio,
            restrict_supplier_positions_tech=rs_tech,
            restrict_consumer_positions=rc_cons,
        )
        lca.apply_strategies()
        lca.evaluate_cfs()

        # Each method yields a 2D plane (m x n). Ensure SciPy CSR, then convert to pydata/sparse COO.
        plane = lca.characterization_matrix
        if not issparse(plane):
            plane = csr_matrix(plane)
        else:
            plane = plane.tocsr()

        planes.append(spnd.COO.from_scipy_sparse(plane))
        edges_lcas[method] = lca

    if not planes:
        # Return an empty 3D tensor of shape (0, 0, 0)
        return spnd.COO(np.zeros((0, 0, 0))).astype(float), multilca_obj, edges_lcas

    # Stack along a NEW leading axis -> (n_methods, m, n)
    characterization_tensor = spnd.stack(planes, axis=0)
    return characterization_tensor, multilca_obj, edges_lcas
