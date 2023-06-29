from typing import List, Tuple

from rdkit import Chem, RDLogger
import torch_geometric
from torch_geometric.data import Data
import torch

from molecular_gcn.data.maps import atomic_map, bond_map

RDLogger.DisableLog("rdApp.*")


class MoleculeEncoder:
    """
    Class that takes a SMILES string and encodes it into a PyTorch Geometric Data object.
    """

    def __init__(self, molecule: str) -> None:
        """
        Constructor for the MoleculeEncoder class.

        Args:
            molecule (str): SMILES string of the molecule to be encoded.
        """
        self.smiles = molecule
        self.mol = Chem.AddHs(Chem.MolFromSmiles(self.smiles))

    def to_data(self) -> torch_geometric.data.Data:
        """
        Method that encodes the molecule into a PyTorch Geometric Data object.

        Returns:
            torch_geometric.data.Data: Data object containing the molecule's
            features.
        """
        x = self._get_node_feature_matrix()
        edge_index, edge_attr = self._get_edge_features(x)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def _get_node_feature_matrix(self) -> torch.Tensor:
        """
        Helper method that encodes the node features of the molecule.

        Returns: 
            torch.Tensor: Node feature matrix of the molecule.
        """
        NUM_NODE_FEATURES = 9

        all_atom_features = []
        for atom in self.mol.GetAtoms():
            atom_features = []
            atom_features.append(atomic_map["atomic_num"].index(
                atom.GetAtomicNum()))
            atom_features.append(atomic_map["chirality"].index(
                str(atom.GetChiralTag())))
            atom_features.append(atomic_map["degree"].index(
                atom.GetTotalDegree()))
            atom_features.append(atomic_map["formal_charge"].index(
                atom.GetFormalCharge()))
            atom_features.append(atomic_map["num_hs"].index(
                atom.GetTotalNumHs()))
            atom_features.append(atomic_map["num_radical_electrons"].index(
                atom.GetNumRadicalElectrons()))
            atom_features.append(atomic_map["hybridization"].index(
                str(atom.GetHybridization())))
            atom_features.append(atomic_map["is_aromatic"].index(
                atom.GetIsAromatic()))
            atom_features.append(atomic_map["is_in_ring"].index(
                atom.IsInRing()))
            all_atom_features.append(atom_features)

        return torch.tensor(all_atom_features,
                            dtype=torch.long).view(-1, NUM_NODE_FEATURES)

    def _get_edge_features(
            self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper method that encodes the edge features of the molecule.

        Args:
            x (torch.Tensor): Node feature matrix of the molecule.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Edge index and edge attribute
            matrices of the molecule.
        """
        edge_indices, edge_attrs = [], []
        for bond in self.mol.GetBonds():
            atom_i = bond.GetBeginAtomIdx()
            atom_j = bond.GetEndAtomIdx()

            edge = []
            edge.append(bond_map["bond_type"].index(str(bond.GetBondType())))
            edge.append(bond_map["stereo"].index(str(bond.GetStereo())))
            edge.append(bond_map["is_conjugated"].index(
                bond.GetIsConjugated()))

            edge_indices += [[atom_i, atom_j], [atom_j, atom_i]]
            edge_attrs += [edge, edge]

        edge_index = torch.tensor(edge_indices)  # (num_edges, 2)
        edge_index = edge_index.t().to(torch.long).view(2,
                                                        -1)  # (2, num_edges)
        edge_attr = torch.tensor(edge_attrs,
                                 dtype=torch.long).view(-1,
                                                        3)  # (num_edges, 3)

        if edge_index.numel() > 0:  # sort edges
            perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

        return (edge_index, edge_attr)


class DataDecoder:
    """
    Class that takes a PyTorch Geometric Data object and decodes it into a SMILES string.
    """

    def __init__(self, data: torch_geometric.data.Data) -> None:
        """
        Constructor for the DataDecoder class.

        Args:
            data (torch_geometric.data.Data): Data object containing the molecule's
            features.
        """
        self.data = data

    def to_smiles(self) -> str:
        """
        Takes the PyTorch Geometric Data object and decodes it into a SMILES string.

        Returns:
            str: SMILES string of the molecule.
        """
        mol = Chem.RWMol()

        for i in range(self.data.num_nodes):
            atom = Chem.Atom(self.data.x[i, 0].item())
            atom.SetChiralTag(
                Chem.rdchem.ChiralType.values[self.data.x[i, 1].item()])
            atom.SetFormalCharge(
                atomic_map['formal_charge'][self.data.x[i, 3].item()])
            atom.SetNumExplicitHs(atomic_map['num_hs'][self.data.x[i,
                                                                   4].item()])
            atom.SetNumRadicalElectrons(
                atomic_map['num_radical_electrons'][self.data.x[i, 5].item()])
            atom.SetHybridization(
                Chem.rdchem.HybridizationType.values[self.data.x[i, 6].item()])
            atom.SetIsAromatic(self.data.x[i, 7].item())
            mol.AddAtom(atom)

        edges = [tuple(i) for i in self.data.edge_index.t().tolist()]
        visited = set()

        for i in range(len(edges)):
            src, dst = edges[i]
            if tuple(sorted(edges[i])) in visited:
                continue

            bond_type = Chem.BondType.values[self.data.edge_attr[i, 0].item()]
            mol.AddBond(src, dst, bond_type)

            # Set stereochemistry:
            stereo = Chem.rdchem.BondStereo.values[self.data.edge_attr[
                i, 1].item()]
            if stereo != Chem.rdchem.BondStereo.STEREONONE:
                db = mol.GetBondBetweenAtoms(src, dst)
                db.SetStereoAtoms(dst, src)
                db.SetStereo(stereo)

            # Set conjugation:
            is_conjugated = bool(self.data.edge_attr[i, 2].item())
            mol.GetBondBetweenAtoms(src, dst).SetIsConjugated(is_conjugated)

            visited.add(tuple(sorted(edges[i])))

        mol = mol.GetMol()
        
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol)

        return Chem.MolToSmiles(mol, isomericSmiles=True)


if __name__ == "__main__":
    molecule = "CC(=O)OC1=CC=CC=C1C(=O)O"
    num_of_atoms = Chem.MolFromSmiles(molecule).GetNumAtoms()
    mol = Chem.MolFromSmiles(molecule)
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        atom_i = bond.GetBeginAtomIdx()
        atom_j = bond.GetEndAtomIdx()

        edge = []
        edge.append(bond_map["bond_type"].index(str(bond.GetBondType())))
        edge.append(bond_map["stereo"].index(str(bond.GetStereo())))
        edge.append(bond_map["is_conjugated"].index(bond.GetIsConjugated()))

        edge_indices += [[atom_i, atom_j], [atom_j, atom_i]]
        edge_attrs += [edge, edge]

    edge_index = torch.tensor(edge_indices)  # (num_edges, 2)
    edge_index = edge_index.t().to(torch.long).view(2, -1)  # (2, num_edges)
    edge_attr = torch.tensor(edge_attrs,
                             dtype=torch.long).view(-1, 3)  # (num_edges, 3)
    print(edge_index)
    print("------")
    print(edge_index[0] * num_of_atoms + edge_index[1])
    print("------")
    print((edge_index[0] * num_of_atoms + edge_index[1]).argsort())
    print("------")
    perm = (edge_index[0] * num_of_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]
    print(edge_index)
    print(edge_attr)