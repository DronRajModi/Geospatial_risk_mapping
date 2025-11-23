import os
import sys
import pandas as pd
import geopandas as gpd
import networkx as nx
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx

from src.exception import CustomException
from src.logger import logger as logging

@dataclass
class GraphFeaturesConfig:
    shapefile_path: str = os.path.join('notebook', 'india_districts.shp')
    imputed_features_path: str = os.path.join('artifacts', 'imputed_features.csv')
    embedding_output_path: str = os.path.join('artifacts', 'gnn_embeddings.csv')

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GraphFeatureGenerator:
    def __init__(self, embedding_dim=32, epochs=50):
        self.config = GraphFeaturesConfig()
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"GNN (Stage C) will run on {self.device}")

    def _build_graph_from_shapefile(self, gdf):
        logging.info("Building spatial adjacency graph from shapefile...")
        adjacency_list = gdf.geometry.touches(gdf.geometry)
        
        g = nx.Graph()
        for i, neighbors in enumerate(adjacency_list):
            if not neighbors:
                g.add_node(i)
                continue
            for neighbor_idx in np.where(neighbors)[0]:
                if neighbor_idx > i:
                    g.add_edge(i, neighbor_idx)
        
        logging.info(f"Graph created with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.")
        return g

    def initiate_graph_feature_generation(self, shapefile_district_col='NAME_2'):
        try:
            logging.info("Starting Stage C: Graph Feature Generation")

            logging.info(f"Loading full map from {self.config.shapefile_path}")
            gdf = gpd.read_file(self.config.shapefile_path)
            
            gdf['District'] = gdf[shapefile_district_col].str.strip().str.title()
            gdf = gdf.set_index('District', drop=False)

            logging.info(f"Loading imputed features from {self.config.imputed_features_path}")
            df_features = pd.read_csv(self.config.imputed_features_path)
            df_features = df_features.set_index('District', drop=False)

            df_features = df_features.reindex(gdf.index)
            
            gdf_for_graph = gdf.reset_index(drop=True)
            G = self._build_graph_from_shapefile(gdf_for_graph)
            
            logging.info("Preparing data for PyTorch Geometric")
            node_features_df = df_features.drop(columns=['District'])
            node_features = node_features_df.values.astype(np.float32)
            
            if np.isnan(node_features).any():
                logging.warning("NaNs found in imputed features. Filling with 0 before GNN training.")
                node_features = np.nan_to_num(node_features)

            x = torch.tensor(node_features, dtype=torch.float)
            
            pyg_data = from_networkx(G)
            edge_index = pyg_data.edge_index
            
            data = Data(x=x, edge_index=edge_index).to(self.device)
            
            logging.info("Defining GNN model...")
            in_channels = data.num_node_features
            model = GraphSAGEModel(
                in_channels=in_channels,
                hidden_channels=64,
                out_channels=self.embedding_dim
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            logging.info(f"Starting GNN training for {self.epochs} epochs...")
            model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                z = model(data)
                loss = F.mse_loss(z, data.x) 
                loss.backward()
                optimizer.step()
                if (epoch + 1) % 10 == 0:
                    logging.info(f'GNN Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}')

            logging.info("Extracting final node embeddings...")
            model.eval()
            with torch.no_grad():
                final_embeddings = model(data).cpu().numpy()
            
            logging.info(f"Saving embeddings to {self.config.embedding_output_path}")
            embed_cols = [f'gnn_embed_{i}' for i in range(self.embedding_dim)]
            df_embeddings = pd.DataFrame(final_embeddings, columns=embed_cols)
            
            df_embeddings['District'] = gdf.index.values
            
            df_embeddings = df_embeddings[['District'] + embed_cols]
            
            os.makedirs(os.path.dirname(self.config.embedding_output_path), exist_ok=True)
            df_embeddings.to_csv(self.config.embedding_output_path, index=False)
            
            logging.info("Stage C: Graph Feature Generation Complete")
            return self.config.embedding_output_path

        except Exception as e:
            raise CustomException(e, sys)