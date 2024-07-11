from matplotlib import pyplot as plt
import numpy as np
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


graph = {
    'nodes': list(range(1, 22)),
    'edges': [
        (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (5, 7), (7, 8), (7, 9),
        (9, 10), (9, 11), (11, 13), (12, 13), (13, 14), (11, 15), (15, 17), (15, 18), 
        (18, 19), (16, 18), (7, 16), (4, 16), (4, 21), (20, 21), (18, 21), (11, 16)
    ],
    'edge_lengths': {
        (1, 2): 1, (2, 3): 2, (2, 4): 5, (4, 5): 15, (5, 6): 7, (5, 7): 13, (7, 8): 8, (7, 9): 14,
        (9, 10): 9, (9, 11): 17, (11, 13): 6, (12, 13): 3, (13, 14): 4, (11, 15): 22, (15, 17): 12, (15, 18): 24, 
        (18, 19): 11, (16, 18): 21, (7, 16): 16, (4, 16): 18, (4, 21): 20, (20, 21): 10, (18, 21): 23, (11, 16): 19
    },
    'edge_indexes': {
        (1, 2): 1, (2, 3): 2, (2, 4): 3, (4, 5): 4, (5, 6): 5, (5, 7): 6, (7, 8): 7, (7, 9): 8,
        (9, 10): 9, (9, 11): 10, (11, 13): 11, (12, 13): 12, (13, 14): 13, (11, 15): 14, (15, 17): 15, (15, 18): 16, 
        (18, 19): 17, (16, 18): 18, (7, 16): 19, (4, 16): 20, (4, 21): 21, (20, 21): 22, (18, 21): 23, (11, 16): 24
    }
}




def simulacija_voznje_vozila(graph, vehicles, num_steps=100):    #num_steps je vremenski okvir u kojemu se mjeri, trenutno traje 50 jedinica
    edge_loads = {}                                             #(trenutak, brid) : broj vozila
    print("pokretanje...")
    for t in range(1, num_steps + 1):                           #u svakom trenutku
        for vehicle in vehicles:                                #za svako vozilo
            route = vehicle["route"]
            speed = vehicle["speed"]                            # speed jedinica duljine brida / 1 jedinica vremena 
            
            distance_traveled = t * speed                       #koliko je vozilo prešlo do ovog trenutka (vrijeme * brzina)

            initial_position = vehicle["initial_position"]      #početna pozicija vozila
            #urediti route ukoliko se ne kreće s početnog čvora
            route = route[route.index(initial_position):] + route[1:route.index(initial_position)+1]    
            
            previous_node = route[0]

            prijedeni_put = 0
            while prijedeni_put < distance_traveled:                #sve dok u petlji nismo simulirali do trenutne situacije
                
                for node in route[1:]:                              #za svaki čvor u route
                    edge = tuple(sorted((previous_node, node)))     #napravi brid
                    edge_length = graph["edge_lengths"][edge]       #duljina trenutnog brida
                    prijedeni_put += edge_length                    #povecaj izracunati put 
                    
                    if prijedeni_put > distance_traveled:           #ako se u simulaciji prešla udaljenost koju je vozilo stvarno prešlo
                        
                        if (t, graph["edge_indexes"][edge]) in edge_loads.keys():
                            edge_loads[t, graph["edge_indexes"][edge]] += 1     #povećati vrijednost za broj vozila na bridu edge u trenutku t
                        else:
                            edge_loads[t, graph["edge_indexes"][edge]] = 1
                        break
                
                    previous_node = node
    
    grouped_data = {}       #brid : (suma, count)

    for key, value in edge_loads.items():
        brid = key[1]
        if brid in grouped_data:
            grouped_data[brid]["sum"] += value
            grouped_data[brid]["count"] += 1
        else:
            grouped_data[brid] = {"sum": value, "count": 1}
    
    averages = {i: 0 for i in range(1, 25)}         #key je index brida, value je prosječan broj vozila

    for key in grouped_data:
        sum_value = grouped_data[key]["sum"]
        count_value = grouped_data[key]["count"]
        
        average = sum_value / count_value           #izračun prosjeka
        
        averages[key] = average
    
    return averages


class GCNVehicleTraffic(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GCNVehicleTraffic, self).__init__()
        self.convs = nn.ModuleList()                            #lista slojeva konvolucijske mreže
        
        self.convs.append(GCNConv(input_dim, hidden_dim))       #dodavanje prvog sloja
        for _ in range(num_layers - 1):                         #dodavanje svih ostalih skrivenih slojeva
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        #linearni sloj
        self.lin = nn.Linear(hidden_dim, 24)                    #dodavanje izlaznog sloja koji ima izlaz od 24 podatka za 24 brida

    def forward(self, data):                                    #definiranje metode za prolaz kroz mrežu
        x, edge_index = data.x, data.edge_index                #ulazni podaci i bridovi
        
        for conv in self.convs:                                 #za svaki sloj
            x = F.elu(conv(x, edge_index))                      #definiranje prijenosne/aktivacijske funkcije - Exponential Linear Unit
            x = F.dropout(x, p=0.1, training=self.training)     #dropout sloj s vjerojatnošću od 0.1 za regularizaciju
        
        x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long))       #pooling koji koristi srednju vrijednost
        out = self.lin(x)                                       #stvaranje izlaznog linearnog sloja
        
        return out.squeeze()


def create_dataloader_from_vehicle_configs(file, graph):
    with open(file, 'r') as f:
        vehicle_configs = json.load(f)  #učitana cijela datoteka
    
    data_list = []
    i = 0
    while (i < len(vehicle_configs)):
        avg_edge_loads = simulacija_voznje_vozila(graph, vehicle_configs[i : i + 10])
        #print("avg loads ", avg_edge_loads)
        edge_index = []
        for edge in graph["edges"]:
            edge_index.append([edge[0]-1, edge[1]-1])
            edge_index.append([edge[1]-1, edge[0]-1])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        #print(edge_index)
        x = torch.tensor(np.zeros((len(graph["nodes"]), 1)), dtype=torch.float)
        y = torch.tensor(list(avg_edge_loads.values()), dtype=torch.float)
        #print("x", x)
        #print("y", y)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
        #print(data_list)

        i += 10     #povećaj da radi sa sljedecim vozilima
    return DataLoader(data_list)



#treniranje modela

def train_model(model, optimizer, dataloader, epochs=1000):   
    
    model.train()                           #postavljanje modela u trening mod
    criterion = nn.MSELoss()                #gubitak (loss) računa se kao Mean Squared Error, srednje kvadratno odstupanje
   
    for epoch in range(epochs):             #za svaku epohu
        total_loss = 0
        for data in dataloader:
            optimizer.zero_grad()               #resetiranje gradijenta
            output = model(data)                #prosljeđivanje podataka kroz model
            #print("output u trainu", output)
            #print("data.y u trainu", data.y)
            loss = criterion(output, data.y)    #izračunavanje gubitka
            loss.backward()                     #algoritam propagacije unatrag
            optimizer.step()                    #ažuriranje parametara modela
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        
        #print(f'Epoch {epoch+1}, Loss {avg_loss}')



def test_model(model, dataloader):
   
    model.eval()                        #postavljanje modela u mod za evaluaciju
    correct_predictions = 0             #brojač točnih predikcija
    total_predictions = 0               #ukupni broj predikcija

    with torch.no_grad():
        for data in dataloader:
            
            output = model(data)
            #print("output u testu ", output)
            #print("data.y u testu ", data.y)

            true_values = data.y      
            predictions = output
            #print("output ", len(predictions))
            #print("true values ", len(true_values))
            
            print("Predikcije za prosječno opterećenje bridova s novim vozilima:")
            for i, (true_value, prediction) in enumerate(zip(true_values, predictions)):
                
                print(f"Brid {i+1}: Stvarno: {true_value:.7f}, Predikcija: {prediction:.7f}")
                
                if true_value != 0:
                    if abs(prediction - true_value) / true_value <= 0.20:
                        correct_predictions += 1
                else:
                    if abs(prediction - true_value) <= 0.20:
                        correct_predictions += 1
                
                total_predictions += 1
    
            accuracy = correct_predictions / total_predictions * 100
            print(f"Točnost: {accuracy:.2f}%")



model = GCNVehicleTraffic(input_dim=1, hidden_dim=32, num_layers=4)     #inicijalizacija modela

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)               #definiranje optimizatora

train_data = create_dataloader_from_vehicle_configs("vehicle_configs_train1_20.json", graph)
test_data = create_dataloader_from_vehicle_configs("vehicle_configs_test1_20.json", graph)

train_model(model, optimizer, train_data)

test_model(model, test_data)          #pokretanje testiranja modela
