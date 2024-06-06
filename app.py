import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import dgl
from sklearn.model_selection import train_test_split
import networkx as nx
from sklearn.preprocessing import PolynomialFeatures
import joblib
import seaborn as sns
st.title('Demand Forecasting with Graph Visualization')

class DemandForecastingApp:
    def __init__(self, loss_fn, model):
        self.timestamp_graphs = {}
        self.loss_fn = loss_fn
        self.model = model
        self.train_graphs = []
        self.test_graphs = []
        self.train_labels = []
        self.test_labels = []
        self.forecasted_graph = []

    def load_data(self, nodes_file, edges_file):
        self.nodes = pd.read_csv(nodes_file)
        self.edges = pd.read_csv(edges_file)
        self.edges = self.edges.drop(columns=['Unnamed: 0'])
        self.filtered_edges = self.edges[self.edges['value'] == 1]

    def visualize_graph(self):
        st.header('Graph Visualization')
        G = nx.from_pandas_edgelist(self.filtered_edges, source='source', target='target', edge_attr=True, create_using=nx.DiGraph())
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, arrowstyle='-|>', arrowsize=20, font_size=10)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(row['source'], row['target']): row['feature'] for idx, row in self.filtered_edges.iterrows()})
        plt.title('Graph Visualization')
        st.pyplot(plt)

    def create_graphs(self):
        src = self.filtered_edges['source'].to_numpy()
        dst = self.filtered_edges['target'].to_numpy()

        for timestamp, group in self.nodes.groupby('timestamp'):
            feat1 = group[group['feature'] == 0].set_index('node')['value']
            feat2 = group[group['feature'] == 1].set_index('node')['value']
            feat3 = group[group['feature'] == 2].set_index('node')['value']
            feat4 = group[group['feature'] == 3].set_index('node')['value']

            g = dgl.graph((src, dst))

            num_nodes = g.num_nodes()
            features = np.zeros((num_nodes, 4))
            features[feat1.index, 0] = feat1
            features[feat2.index, 1] = feat2
            features[feat3.index, 2] = feat3
            features[feat4.index, 3] = feat4
            features = torch.FloatTensor(features)

            g.ndata['features'] = features
            self.timestamp_graphs[timestamp] = g
        self.split_graphs(0.8)

    def split_graphs(self, train_size=0.8):
        timestamps = sorted(self.timestamp_graphs.keys(), key=lambda x: x)
        train_timestamps = timestamps[0:int(train_size * len(timestamps))]
        test_timestamps = timestamps[int(train_size * len(timestamps)):]

        self.train_graphs = []
        self.test_graphs = []
        self.train_labels = []
        self.test_labels = []

        for timestamp in train_timestamps:
            self.train_graphs.append(self.timestamp_graphs[timestamp])
            self.train_labels.append(self.timestamp_graphs[timestamp].ndata['features'][:, 3].unsqueeze(0))

        for timestamp in test_timestamps:
            self.test_graphs.append(self.timestamp_graphs[timestamp])
            self.test_labels.append(self.timestamp_graphs[timestamp].ndata['features'][:, 3].unsqueeze(0))

    def train_model(self, epochs):
        st.header('Training the Model')
        train_graphs, train_labels = self.train_graphs, self.train_labels

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for i in range(len(train_graphs)):
                g = train_graphs[i]
                features = g.ndata['features'][:, :3]  # Use the first three features
                labels = train_labels[i]

                optimizer.zero_grad()
                predictions = self.model(g, features).squeeze()
                loss = self.loss_fn(predictions, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            st.write(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_graphs)}')

        self.model.eval()
        with torch.no_grad():
            st.title('Sample Graph')
            test_graph = self.train_graphs[-1]
            test_features = test_graph.ndata['features'][:, :3]
            predictions = self.model(test_graph, test_features).squeeze()

            # Get actual values
            actual_values = test_graph.ndata['features'][:, -1]
            print("Predictions:", predictions.numpy())
            print("Actual values:", actual_values.numpy())

            predictions_np = predictions.numpy()
            actual_values_np = actual_values.numpy()

            # Ensure actual_values_np is a one-dimensional array
            if actual_values_np.ndim > 1:
                actual_values_np = actual_values_np.flatten()

            plt.figure(figsize=(10, 6))
            x = np.arange(len(predictions_np))
            width = 0.35
            plt.bar(x - width/2, predictions_np, width, label='Predictions')
            plt.bar(x + width/2, actual_values_np, width, label='Actual Values')
            plt.xlabel('Node Index')
            plt.ylabel('Feature Value')
            plt.title('Predictions vs Actual Values')
            plt.legend()
            st.pyplot(plt)

    def evaluate_model(self):
        test_graphs = self.test_graphs
        st.header('Model Evaluation')

        self.model.eval()
        self.all_predictions = []
        all_actual_values = []
        all_losses = []

        with torch.no_grad():
            for g in test_graphs:
                features = g.ndata['features'][:, :3]
                labels = g.ndata['features'][:, -1]

                predictions = self.model(g, features).squeeze()
                loss = self.loss_fn(predictions, labels)
                self.all_predictions.append(predictions.numpy())
                all_actual_values.append(labels.numpy())
                all_losses.append(loss.item())

        # Evaluation Section 1: Loss Curve
        st.subheader('Loss Curve over Test Set')
        plt.figure(figsize=(10, 5))
        plt.plot(all_losses, label='Loss', color='blue')
        plt.xlabel('Graph Index')
        plt.ylabel('Loss')
        plt.legend()
        st.pyplot(plt)

        # Evaluation Section 2: Predictions vs Actual Values for Each Graph
        num_graphs = len(self.all_predictions)
        plt.figure(figsize=(15, 5 * num_graphs))

        for i in range(num_graphs):
            plt.subplot(num_graphs, 1, i + 1)
            x = np.arange(len(self.all_predictions[i]))
            width = 0.35

            actual_values_np = np.array(all_actual_values[i])
            if actual_values_np.ndim > 1:
                actual_values_np = actual_values_np.flatten()

            plt.bar(x - width/2, self.all_predictions[i], width, label='Predictions')
            plt.bar(x + width/2, actual_values_np, width, label='Actual Values')
            plt.xlabel('Node Index')
            plt.ylabel('Feature Value')
            plt.title(f'Predictions vs Actual Values for Graph {i + 1}')
            plt.legend()

        plt.tight_layout()
        st.pyplot(plt)

        # Evaluation Section 3: Concatenated Predictions vs Actual Values
        st.subheader('Concatenated Predictions vs Actual Values')
        concatenated_predictions = np.concatenate(self.all_predictions)
        concatenated_actual_values = np.concatenate(all_actual_values)

        plt.figure(figsize=(15, 5))
        x = np.arange(len(concatenated_predictions[:100]))
        width = 0.35
        plt.bar(x - width/2, concatenated_predictions[:100], width, label='Predictions', color='green')
        plt.bar(x + width/2, concatenated_actual_values[:100], width, label='Actual Values', color='orange')
        plt.xlabel('Node Index')
        plt.ylabel('Feature Value')
        plt.title('Concatenated Predictions vs Actual Values')
        plt.legend()
        st.pyplot(plt)

    def generate_random_values(self,predictions):
        random_values = np.random.rand(len(predictions), 3)
        return random_values
    
    # def predict_feature3(self, feature3_tensor):
    #     df = self.nodes
    #     df = df.drop("Unnamed: 0", axis=1)
    #     df_pivoted = df.pivot_table(index=['timestamp', 'node'], columns='feature', values='value')
    #     df_pivoted.columns = ['Feature0', 'Feature1', 'Feature2', 'Feature3']
    #     df_final = df_pivoted.reset_index()
    #     df_final = df_final.drop(['timestamp', 'node'], axis=1)
    #     poly = PolynomialFeatures(degree=2, include_bias=False)
    #     poly.fit_transform(df_final[['Feature3']])
    #     model = joblib.load('best_model.pkl')

    #     # Convert tensor to numpy array
    #     feature3_values = feature3_tensor.numpy().reshape(-1, 1)
    #     # Transform the input features
    #     feature3_poly = poly.transform(feature3_values)

    #     # Predict using the model
    #     predictions = model.predict(feature3_poly)

    #     return np.array(predictions)

    

    def plot_feature3_values(self, node_id):
        # Collect Feature3 values for train, test, and forecasted graphs
        train_feature3 = []
        test_feature3 = []
        forecasted_feature3 = []
        test_preds=[]

        for graph, labels in zip(self.train_graphs, self.train_labels):
            train_feature3.append(graph.ndata['features'][node_id, 2].item())

        for graph, labels in zip(self.test_graphs, self.test_labels):
            test_feature3.append(graph.ndata['features'][node_id, 2].item())

        for graph in self.forecasted_graph:
            forecasted_feature3.append(graph.ndata['features'][node_id, 2].item())
        for i in self.all_predictions:
            test_preds.append(i[node_id])

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=range(len(train_feature3)), y=train_feature3, label='Train', color='blue')
        sns.scatterplot(x=range(len(train_feature3), len(train_feature3) + len(test_feature3)), y=test_feature3, label='Test', color='orange')
        sns.scatterplot(x=range(len(train_feature3) + len(test_feature3), len(train_feature3) + len(test_feature3) + len(forecasted_feature3)), y=forecasted_feature3, label='Forecasted', color='green')
        sns.scatterplot(x=range(len(train_feature3), len(train_feature3) + len(test_feature3)), y=test_preds, label='Predicted Test', color='red')
        plt.xlabel('Timestamp')
        plt.ylabel('Feature3 Value')
        plt.title(f'Feature3 Values for Node {node_id}')
        plt.legend()
        st.pyplot(plt)

    def forecasting(self):
        st.header('Forecasting')
        latest = max(self.timestamp_graphs.keys())
        latest_graph = self.timestamp_graphs[latest]

        limit = st.number_input('Enter number of timestamps to forecast:', min_value=1, value=10, step=1)
        
        for i in range(limit):
            features = latest_graph.ndata['features'][:, :3]
            with torch.no_grad():
                predictions = self.model.forward(latest_graph, features).squeeze()
                feature3_predictions = self.generate_random_values(predictions)
                
                # Convert the feature3_predictions to tensor
                feature3_values_tensor = torch.tensor(feature3_predictions, dtype=torch.float32)

                # Ensure predictions_tensor is of correct shape
                predictions_tensor = predictions.unsqueeze(1)

                # Concatenate the new feature tensors
                new_features = torch.cat((feature3_values_tensor, predictions_tensor), dim=1)

                # Create a new graph with the same structure as the latest graph but updated features
                new_graph = dgl.graph(latest_graph.edges())
                new_graph.ndata['features'] = new_features

                self.forecasted_graph.append(new_graph)
                latest_graph = new_graph
        
        st.write(f"Forecasted {limit} new graphs.")

        # Input slider for specifying the node_id
        # node_id = st.slider('Select Node ID:', min_value=0, max_value=self.timestamp_graphs[0].num_nodes() - 1, value=0)
        for node_id in range (self.timestamp_graphs[0].num_nodes()):
        # Plot Feature3 values for the specified node_id
            self.plot_feature3_values(node_id)
        



class TemporalGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers):
        super(TemporalGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_feats, hidden_feats))
        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_feats, hidden_feats))
        self.rnn = nn.GRU(hidden_feats, hidden_feats, batch_first=True)
        self.fc = nn.Linear(hidden_feats, out_feats)
    
    def forward(self, g, features):
        if g is not None:
            h = features
            for conv in self.convs:
                h = F.relu(conv(g, h))
        else:
            # Assume features are already processed and can be used directly
            h = features
        
        h = h.unsqueeze(1)  # Add time dimension
        out, _ = self.rnn(h)
        out = self.fc(out[:, -1, :])
        return out

    def predict(self, features):
        # Prediction mode without graph
        return self.forward(None, features)

# Sidebar widgets for parameter selection
loss_functions = {
    "Mean Squared Error": nn.MSELoss(),
    # Add more loss functions here if needed
}

model_options = {
    "TemporalGNN": TemporalGNN,
    # Add more models here if needed
}

selected_loss = st.sidebar.selectbox("Select Loss Function", list(loss_functions.keys()))
selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()))

# Additional hyperparameters
in_feats = st.sidebar.slider("Input Features", min_value=1, max_value=10, value=3)
hidden_feats = st.sidebar.slider("Hidden Features", min_value=16, max_value=128, value=64)
out_feats = st.sidebar.slider("Output Features", min_value=1, max_value=10, value=1)
num_layers = st.sidebar.slider("Number of Layers", min_value=1, max_value=5, value=2)

loss_fn = loss_functions[selected_loss]
model_class = model_options[selected_model]

# Instantiate DemandForecastingApp with selected parameters
app = DemandForecastingApp(loss_fn, model_class(in_feats, hidden_feats, out_feats, num_layers))

# Upload nodes and edges CSV files
uploaded_nodes_file = st.file_uploader('Upload Nodes CSV', type='csv')
uploaded_edges_file = st.file_uploader('Upload Edges CSV', type='csv')

if uploaded_nodes_file and uploaded_edges_file:
    app.load_data(uploaded_nodes_file, uploaded_edges_file)

    st.write('Nodes DataFrame:')
    st.dataframe(app.nodes)

    st.write('Edges DataFrame:')
    st.dataframe(app.filtered_edges)

    app.visualize_graph()
    app.create_graphs()
    
    # Input for number of epochs
    epochs = st.number_input('Enter number of epochs:', min_value=1, value=5)
    
    if st.button('Begin'):
        app.train_model(epochs)
        # if i create a seperate buttons, a different instance of model is being used giving innacurate prediction_BUG
        app.evaluate_model()
        app.forecasting()
else:
    st.write('Please upload both Nodes and Edges CSV files to proceed.')
