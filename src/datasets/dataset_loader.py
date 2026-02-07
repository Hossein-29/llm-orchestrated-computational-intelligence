import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from sklearn.datasets import make_blobs
from src.utils.tsp_solver import solve_tsp_lkh,solve_tsp_exact

DATASET_NAMES=["eil51","berlin52","kroa100","iris","mall customers"]

def dataset_loader(problem_info:dict):

  if problem_info["problem_type"].lower()=="tsp":
    return tsp_dataset_loader(problem_info)
  
  elif problem_info["problem_type"].lower()=="clustering":
    return clustering_dataset_loader(problem_info)

def tsp_dataset_loader(problem_info:dict):

  source = problem_info["content"]["data_source"]

  try:

    if source.lower() in DATASET_NAMES:
        with open(f"./src/datasets/{source.lower()}.tsp") as f:
          lines=f.readlines()
        cities_cords=[]
        for line in lines:
          if line=="EOF":
            break
          else:
            try:
              _,x,y=map(float,line.split(" "))
              cities_cords.append([x,y])
            except:
              continue
        cities_cords=np.array(cities_cords)
        n_cities= cities_cords.shape[0]
        distance_matrix=cdist(cities_cords,cities_cords,metric="euclidean")
        problem_info["content"]["n_cities"]=n_cities
        return distance_matrix
        
    elif source.lower() == "generated":
      n_cities=problem_info["content"]["n_cities"]
      cities_cords=np.random.rand(n_cities, 2) * 100
      distance_matrix=cdist(cities_cords,cities_cords,metric="euclidean")
      # if n_cities==30:
      #   optimal_path=solve_tsp_exact(distance_matrix)
      # else:
      #   optimal_path=solve_tsp_lkh(distance_matrix)
      # problem_info["content"]["known_optimal"]=optimal_path
      return distance_matrix
        
  except:
    return source


def clustering_dataset_loader(problem_info:dict):
  source = problem_info["content"]["data_source"]

  try:
    if source.lower() in DATASET_NAMES:
        if source == "iris":
          df = pd.read_csv("src/problems/datasets/iris.csv")
          return df
        
        elif source == "mall customer":
          df = pd.read_csv("src/problems/datasets/Mall_Customers.csv")
          return df
        
        else:
            return pd.read_csv(f"src/problems/datasets/{source.lower()}.csv")
        
    elif source.lower() == "generated":
      
      n_samples=problem_info["content"]["n_samples"] if problem_info["content"]["n_samples"] != "not_specified" else 100,
      n_clusters=problem_info["content"]["n_clusters"] if problem_info["content"]["n_clusters"] != "not_specified" else 2,
      n_features=problem_info["content"]["n_features"] if problem_info["content"]["n_features"] != "not_specified" else 2,
      cluster_std=problem_info["content"]["cluster_std"] if problem_info["content"]["cluster_std"] != "not_specified" else 1

      X, y = make_blobs(
          n_samples=n_samples,
          centers=n_clusters,
          n_features=n_features,
          cluster_std=cluster_std,
          random_state=42
      )
      columns = [f"Feature_{i+1}" for i in range(n_features)]
      df = pd.DataFrame(X, columns=columns)

      df["target"] = y
      
      return df
        
  except:
    return pd.DataFrame(source)