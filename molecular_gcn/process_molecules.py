from typing import List

from rdkit import Chem, RDLogger
import torch_geometric
from molecular_gcn.maps import atomic_map, bond_map

RDLogger.DisableLog('rdApp.*')


class MoleculeHandler:
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/smiles.html

    def __init__(self, molecule: str):
        self.smiles = molecule
        self.mol = Chem.AddHs(Chem.MolFromSmiles(self.smiles))

    def to_data(self) -> torch_geometric.data.Data:
        pass

    def _get_adjacency_matrix(self):
        return Chem.GetAdjacencyMatrix(self.mol)

    def _get_atoms(self):
        return list(self.mol.GetAtoms())


if __name__ == "__main__":
    mol_handler = MoleculeHandler("CCC")
    print(mol_handler._get_adjacency_matrix())