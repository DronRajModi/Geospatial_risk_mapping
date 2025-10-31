# src/components/graph_features.py

import os
import sys
import pandas as pd
import geopandas as gpd
import networkx as nx
import torch
import torch.nn.functional as F
from dataclasses import dataclass

# --- NEW IMPORTS ---
# We use torch_geometric for the GNN
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx
# ---

from src.exception import CustomException
from src.logger import logger as logging

@dataclass
class GraphFeaturesConfig:
    shapefile_path: str = os.path.join('notebook', 'india_districts.shp')
    imputed_features_path: str = os.path.join('artifacts', 'imputed_features.csv')
    embedding_output_path: str = os.path.join('artifacts', 'gnn_embeddings.csv')

# --- Define the GNN Model (GraphSAGE) ---
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModel, self).__init__()
        # 2-layer GraphSAGE model
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        # We return the embeddings before the final activation
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
        # Create an adjacency list where each district (row) maps to the indices
        # of the districts it touches.
        adjacency_list = gdf.geometry.touches(gdf.geometry)
        
        # Convert to a NetworkX graph
        g = nx.Graph()
        for i, neighbors in enumerate(adjacency_list):
            if not neighbors: # Add node even if it has no neighbors
                g.add_node(i)
                continue
            for neighbor_idx in np.where(neighbors)[0]: # Get indices of neighbors
                if neighbor_idx > i: # Avoid duplicate edges
                    g.add_edge(i, neighbor_idx)
        
        logging.info(f"Graph created with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.")
        return g

    def initiate_graph_feature_generation(self, shapefile_district_col='NAME_2'):
        try:
            logging.info("Starting Stage C: Graph Feature Generation")

            # 1. Load Map (The Master List of Districts)
            logging.info(f"Loading full map from {self.config.shapefile_path}")
            gdf = gpd.read_file(self.config.shapefile_path)
            
            # Standardize district names for merging
            gdf['District'] = gdf[shapefile_district_col].str.strip().str.title()
            gdf = gdf.set_index('District', drop=False) # Use district as index for easy mapping

            # 2. Load Features (The Node Features)
            logging.info(f"Loading imputed features from {self.config.imputed_features_path}")
            df_features = pd.read_csv(self.config.imputed_features_path)
            df_features = df_features.set_index('District', drop=False)

            # Ensure GDF and DF features are in the same order
            # This is critical.
            df_features = df_features.reindex(gdf.index)
            
            # 3. Build Graph
            # Reset index so that GDF index (0..N) matches graph node IDs
            gdf_for_graph = gdf.reset_index(drop=True)
            G = self._build_graph_from_shapefile(gdf_for_graph)
            
            # 4. Prepare Data for PyTorch Geometric
            logging.info("Preparing data for PyTorch Geometric")
            # Get node features from the imputed CSV (already sorted by gdf index)
            # Ensure we only use numeric feature columns
            node_features_df = df_features.drop(columns=['District'])
            node_features = node_features_df.values.astype(np.float32)
            
            # Check for NaNs in features, which can break GNN training
            if np.isnan(node_features).any():
                logging.warning("NaNs found in imputed features. Filling with 0 before GNN training.")
                node_features = np.nan_to_num(node_features)

            # Convert to PyTorch tensor
            x = torch.tensor(node_features, dtype=torch.float)
            
            # Get edge index from NetworkX graph
            pyg_data = from_networkx(G)
            edge_index = pyg_data.edge_index
            
            data = Data(x=x, edge_index=edge_index).to(self.device)
            
            # 5. Define and Train GNN
            logging.info("Defining GNN model...")
            in_channels = data.num_node_features
            model = GraphSAGEModel(
                in_channels=in_channels,
                hidden_channels=64, # 64-dim hidden layer
                out_channels=self.embedding_dim # 32-dim output embedding
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            # We train the GNN to be good at reconstructing the node features.
            # This is an unsupervised task.
            logging.info(f"Starting GNN training for {self.epochs} epochs...")
            model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                z = model(data) # Get embeddings
                # The loss is how well the embeddings can reconstruct the original features
                loss = F.mse_loss(z, data.x) 
                loss.backward()
                optimizer.step()
                if (epoch + 1) % 10 == 0:
                    logging.info(f'GNN Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}')

            # 6. Extract Embeddings
            logging.info("Extracting final node embeddings...")
            model.eval()
            with torch.no_grad():
                final_embeddings = model(data).cpu().numpy()
            
            # 7. Save Embeddings
            logging.info(f"Saving embeddings to {self.config.embedding_output_path}")
            # Create column names for embeddings
            embed_cols = [f'gnn_embed_{i}' for i in range(self.embedding_dim)]
            df_embeddings = pd.DataFrame(final_embeddings, columns=embed_cols)
            
            # Add the District index back
            df_embeddings['District'] = gdf.index.values
            
            # Reorder columns to have District first
            df_embeddings = df_embeddings[['District'] + embed_cols]
            
            os.makedirs(os.path.dirname(self.config.embedding_output_path), exist_ok=True)
            df_embeddings.to_csv(self.config.embedding_output_path, index=False)
            
            logging.info("Stage C: Graph Feature Generation Complete")
            return self.config.embedding_output_path

        except Exception as e:
            raise CustomException(e, sys)