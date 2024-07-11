import json
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class GCNFiedler(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GCNFiedler, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.lin = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))       
            x = F.dropout(x, p=0.1, training=self.training)     
        
        x = global_mean_pool(x, batch)
        out = self.lin(x)
        
        return out.squeeze()

def create_dataloader_from_graphs(json_file, batch_size=32):
    with open(json_file, 'r') as f:     #otvara datoteku za čitanje
        graphs_data = json.load(f)
    
    data_list = []                              #inicijlazicaije liste podataka čiji će elementi bit u Data obliku
    for graph in graphs_data:                   #za svaki graf
        #sve podatke pretvori u tensor oblik
        adjacency_matrix = torch.tensor(graph['adjacency_matrix'], dtype=torch.float)
        node_features = torch.tensor(graph['node_features'], dtype=torch.float).view(-1, 1)
        fiedler_value = torch.tensor([graph['fiedler_value']], dtype=torch.float)
        
        edge_index = adjacency_matrix.nonzero(as_tuple=False).t().contiguous()  #generira indekse bridova iz matrice susjedstva, transponira i osigurava da su poredani u memoriji
        
        data = Data(x=node_features, edge_index=edge_index, y=fiedler_value)        #stvara Data objekt s ucitanim podacima i dodaje ga u listu
        data_list.append(data)
    
    return DataLoader(data_list, batch_size=batch_size, shuffle=True)       #vraća DataLoader koji će učitavati podatke u batcheve određene veličine i na random, neće ići redom kojim su upisani u datoteci

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') maknula sam sve za ovaj device!!!!!!
input_dim = 1
hidden_dim = 64
num_layers = 8           
model = GCNFiedler(input_dim, hidden_dim, num_layers)   #.to(device)


dataloader = create_dataloader_from_graphs('graphs500.json', batch_size=32)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)    
criterion = nn.MSELoss()

errors = []
def train(model, optimizer, criterion, dataloader):
    model.train()
    total_loss = 0
    for data in dataloader:
        #data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        #errors.append(loss.item())
        total_loss += loss.item()
    return total_loss / len(dataloader)

num_epochs = 1500        #5000
for epoch in range(num_epochs):
    loss = train(model, optimizer, criterion, dataloader)
    errors.append(loss)
    print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')

def test(model, dataloader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        graph_index = 1
        for data in dataloader:
            #data = data.to(device)
            output = model(data)
            true_values = data.y.cpu().numpy()
            predictions = output.cpu().numpy()

            for i in range(len(true_values)):
                true_value = true_values[i]
                prediction = predictions[i]

                print(f"Graf {graph_index}: stvarno: {true_value:.3f}, predikcija: {prediction:.3f}")
                graph_index += 1

                if true_value != 0:
                    if abs(prediction - true_value) / true_value <= 0.20:
                        correct_predictions += 1
                else:
                    if abs(prediction - true_value) <= 0.20:
                        correct_predictions += 1

                total_predictions += 1

    accuracy = correct_predictions / total_predictions * 100
    print(f"Točnost: {accuracy:.2f}%")

test_data = create_dataloader_from_graphs('graphs500_test.json', batch_size=20) 
test(model, test_data)
