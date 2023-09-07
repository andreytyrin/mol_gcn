import pandas as pd

from torch_geometric.loader import DataLoader

from molecular_gcn.data.process_molecules import MoleculeEncoder


class MoleculeDataLoader:
    """
    Class that loads the data from the CSV file and creates a DataLoader
    object for the dataset.
    """

    def __init__(self, csv_file: str, batch_size: int):
        """
        Args:
            csv_file (str): Path to the CSV file containing the dataset.
            batch_size (int): Batch size for the DataLoader object.
        """
        self.df = pd.read_csv(csv_file)
        self.batch_size = batch_size
        self._smiles = self.df["SMILES"].tolist()
        self._targets = self.df["Y"].tolist()
        self._data = []
        self._update_data()  # mutates self._data

    def _update_data(self) -> None:
        """
        Method that updates the data attribute of the class with the Data objects 
        that are created from the SMILES strings in the dataset. Mutates self._data.
        """
        for smiles, target in zip(self._smiles, self._targets):
            encoder = MoleculeEncoder(smiles)
            mol_data_object = encoder.to_data()
            mol_data_object.y = target
            self._data.append(mol_data_object)

    def get_data_loader(self) -> DataLoader:
        """
        Method that creates a DataLoader object for the dataset.

        Returns:
            DataLoader: DataLoader object for the dataset.
        """
        return DataLoader(self._data, batch_size=self.batch_size)


if __name__ == "__main__":
    csv_file = "..."
    batch_size = 32
    loader = MoleculeDataLoader(csv_file, batch_size)
    data_loader = loader.get_data_loader()