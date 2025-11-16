# Graph Neural Network for Bike Usage Prediction
*A spatial–temporal forecasting model using Graph Neural Networks (GCN) to predict next-hour bike demand across city stations.*

---

## Overview

This project builds a **Graph Neural Network (GNN)** to forecast bike-sharing demand at each station for the next hour.  
Stations are modelled as **nodes**, and geographic proximity is used to form **edges** between stations.  

The model learns from:
- recent bike demand  
- hour of day  
- weekend/weekday indicator  
- temperature  
- rainfall  
- station connectivity (graph structure)

This mirrors how real bike-sharing systems behave: demand depends on both **local station patterns** and **neighbors in the city**.

---

##  Why Use a Graph Neural Network?

Bike share systems are inherently **spatial**:

- nearby stations influence each other  
- demand spreads through neighbors (e.g., work areas, transit hubs)  
- temporal factors also matter (rush hour, weather, weekend effects)

Traditional fully-connected models ignore this.  
GCNs allow the model to learn these *spatial dependencies* directly from the graph.

---


The notebook contains all steps end-to-end:
- synthetic dataset generation  
- graph construction using k-nearest neighbors  
- GCN model  
- training loop  
- prediction and visualization  

---

## 📊 Dataset (Synthetic but Realistic)

The dataset simulates:
- **15 bike stations** (coordinates similar to Amsterdam)  
- **500 hours** of operation  
- realistic patterns:
  - morning & evening peaks  
  - temperature effects  
  - rainfall decreasing demand  
  - weekend dips  

Each row includes: hour, station_id, lat, lon, hour_of_day, is_weekend, temp, rain, demand


The target is **next-hour demand** for each station.

---

## Graph Construction

Stations are connected using **3-nearest neighbors** based on latitude/longitude.  
This creates an undirected graph:

Nodes = bike stations
Edges = spatial neighbors
Features = [hour_of_day, is_weekend, temp, rain, demand]
Target = next-hour demand


PyTorch Geometric’s `GCNConv` is used to propagate information across the graph.

---

## Model Architecture

A simple 2-layer GCN:

Input (5 features per station)
↓
GCNConv + ReLU (hidden_dim=32)
↓
GCNConv (output_dim=1)
↓
Next-hour demand per station


The model is trained across all time steps (graphs stay the same, features change).

---

##  Training

Loss function: **MSE (Mean Squared Error)**  
Optimizer: **Adam**  
Device: GPU (Colab CUDA) if available  

Training loops through each hour’s graph snapshot:
for each hour t:
X[t] → model → predict_y[t]
target_y[t] → MSE → backprop


---

## Results

After training:

- predicted demand closely follows real patterns  
- model captures rush-hour peaks and weather effects  
- stations with strong spatial correlations show smoother, more accurate predictions  

The notebook includes a plot too
---

## Possible Applications

- Real-time bike redistribution  
- Optimizing rebalancing truck routes  
- Predictive maintenance for high-traffic stations  
- City mobility planning  
- Seasonal or event-based forecasting  

---

##  Future Improvements

You can extend this project with:

- **GAT (Graph Attention Networks)**  
- **GraphSAGE for inductive learning**  
- integrating **real datasets** (Chicago, CitiBike, Bixi)  
- multi-step forecasting (predict next 3 hours)  
- dynamic graphs (changing station connectivity)  

---

## Technologies Used

- Python  
- PyTorch  
- PyTorch Geometric  
- pandas / numpy  
- scikit-learn  
- Google Colab GPU  
- Nearest Neighbors (for graph construction)  

---




