import pandas as pd
import numpy as np
import dgl
import torch
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from scipy.sparse import csr_matrix
from dgl.data import DGLDataset


class CustomGraphDataset(DGLDataset):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        super().__init__(name='custom_graph')

    def process(self):
        # Step 1: Load data
        data = pd.read_csv(self.csv_path)

        # Step 2: Preprocess features
        # Column 1: 区域编码 (One-hot Encoding)
        region_encoder = OneHotEncoder(sparse=False)
        region_features = region_encoder.fit_transform(data.iloc[:, 0].values.reshape(-1, 1))

        # Column 2-19: 栅格类型占比 (Unit Normalization)
        type_features = data.iloc[:, 1:19].values
        type_features = type_features / type_features.sum(axis=1, keepdims=True)

        # Column 20-47: POI数量 (Min-Max Scaling + Log Scaling)
        poi_features = data.iloc[:, 19:47].values
        poi_features = np.log1p(poi_features)  # Log scaling to reduce skewness
        poi_scaler = MinMaxScaler()
        poi_features = poi_scaler.fit_transform(poi_features)

        # Column 48-65: AOI类型占比 (Unit Normalization)
        aoi_features = data.iloc[:, 47:65].values
        aoi_features = aoi_features / aoi_features.sum(axis=1, keepdims=True)

        # Column 66-67: 建筑面积和容积 (Min-Max Scaling + Log Scaling)
        building_features = data.iloc[:, 65:67].values
        building_features = np.log1p(building_features)
        building_scaler = MinMaxScaler()
        building_features = building_scaler.fit_transform(building_features)

        # Column 68-84: 路段长度 (Min-Max Scaling + Log Scaling)
        road_features = data.iloc[:, 67:84].values
        road_features = np.log1p(road_features)
        road_scaler = MinMaxScaler()
        road_features = road_scaler.fit_transform(road_features)

        # Step 3: Concatenate all features
        all_features = np.hstack([
            region_features,
            type_features,
            poi_features,
            aoi_features,
            building_features,
            road_features
        ])
        all_features = torch.tensor(all_features, dtype=torch.float32)

        # Step 4: Create adjacency matrix (example placeholder, needs real adjacency data)
        # This should be replaced with the actual adjacency matrix for your data
        num_nodes = all_features.shape[0]
        adj_matrix = np.eye(num_nodes)  # Using identity matrix as placeholder

        # Convert adjacency matrix to DGL graph
        g = dgl.from_scipy(csr_matrix(adj_matrix))

        # Add node features
        g.ndata['feat'] = all_features

        self.graph = g

    def __getitem__(self, idx):
        return self.graph

    def __len__(self):
        return 1  # Only one graph


# Usage
csv_path = 'your_data.csv'  # Path to your CSV file
dataset = CustomGraphDataset(csv_path)
graph = dataset[0]
print(graph)
