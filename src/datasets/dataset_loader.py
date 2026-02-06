import numpy as np
from scipy.spatial.distance import cdist

DATASET_NAMES=["eil51","berlin52","kroa100","iris"]

def dataset_loader(problem_info:dict):
  if problem_info["content"]["dataset_name"].lower() in DATASET_NAMES:

    if problem_info["problem_type"].lower()=="tsp":
      return tsp_dataset_loader(problem_info)
    
    elif problem_info["problem_type"].lower()=="optimization":
      return optimization_dataset_loader(problem_info)

def tsp_dataset_loader(problem_info:dict):

  with open(f"./src/datasets/{problem_info["content"]["dataset_name"]}.tsp") as f:
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

  problem_info["content"]["distance_matrix"]=distance_matrix
  problem_info["content"]["n_cities"]=n_cities

def optimization_dataset_loader(problem_content:dict):
  pass