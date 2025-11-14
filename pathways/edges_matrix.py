"""
This module defines runs Edges, to produce a (regionalized) characterization matrix.
"""

import bw2calc
import json
from typing import Optional, Dict
from edges import EdgeLCIA
from edges.matrix_builders import build_technosphere_edges_matrix
from edges import setup_package_logging
from bw_processing import Datapackage
from scipy.sparse import csr_matrix, vstack, issparse
import numpy as np
import sparse as spnd
import logging

from .filesystem_constants import DATA_DIR

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


def create_edges_characterization_matrix(
    model: str,
    multilca_obj: bw2calc.MultiLCA,
    methods: list,
    indices: dict[str, dict[int, dict]],
):
    """
    Run Edges for each method and return a 3D sparse tensor of shape
    (n_methods, m, n), where each [i, :, :] is that method's characterization plane.
    """
    topology = fetch_topology(model)

    planes = []

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

    if not planes:
        # Return an empty 3D tensor of shape (0, 0, 0)
        return spnd.COO(np.zeros((0, 0, 0))).astype(float)

    # Stack along a NEW leading axis -> (n_methods, m, n)
    characterization_tensor = spnd.stack(planes, axis=0)
    return characterization_tensor, multilca_obj
