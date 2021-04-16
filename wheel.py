import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from ipywidgets import interact, interactive, widgets
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score
from scipy.stats import mode
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import random
import datetime
from genetic import *
from itertools import chain


gsm = pd.read_csv("https://raw.githubusercontent.com/wesleybeckner/mfg_product_wheel/main/data/gsm_transitions.csv", index_col=0)
gsm.columns = gsm.columns.astype(float)

tech = pd.read_csv("https://raw.githubusercontent.com/wesleybeckner/mfg_product_wheel/main/data/tech_transitions.csv", index_col=0)

gsm_dict = gsm.to_dict()
tech_dict = tech.to_dict()

# to start we need to create some test data (forecasted orders)
random.seed(42)
def my_little_orders_generator(order_count,
                               gsms,
                               techs,
                               kgs):
  orders = []
  for i in range(order_count):
    orders.append([random.choice(gsms),
                  random.choice(techs),
                  random.randint(kgs[0], kgs[1])])
  return orders

orders  = my_little_orders_generator(100,
                           gsm.index,
                           tech.index,
                           [1000,2000])
orders = []
for i in tech.index:
  for j in gsm.index:
    orders.append([j,i,random.randint(1000,2000)])

orders = pd.DataFrame(orders, columns=['gsm', 'tech', 'kg']).to_dict(orient='index')

class Pair:
  def __init__(self, node, adjacent):
    if node < adjacent:
      node, adjacent = adjacent, node
    self.Node = node
    self.Adjacent = adjacent

  def __eq__(self, other):
    return self.Node == other.Node and self.Adjacent == other.Adjacent

  def __hash__(self):
    return hash(self.Node) * 397 ^ hash(self.Adjacent)

def crossover(parentGenes, donorGenes, fnGetFitness):
  # pairs in donor including last-first
  pairs = {Pair(donorGenes[0], donorGenes[-1]): 0}
  for i in range(len(donorGenes) - 1):
    pairs[Pair(donorGenes[i], donorGenes[i+1])] = 0

  # check if there is a place to swap with parent
  tempGenes = parentGenes[:]
  if Pair(parentGenes[0], parentGenes[-1]) in pairs:
    # find a discontinuity
    found = False
    for i in range(len(parentGenes) -1):
      if Pair(parentGenes[i], parentGenes[i+1]) in pairs:
        continue
      tempGenes = parentGenes[i + 1:] + parentGenes[:i + 1]
      found = True
      break
    if not found:
      return None

  # find all the similar runs in both
  runs = [[tempGenes[0]]]
  for i in range(len(tempGenes) - 1):
    if Pair(tempGenes[i], tempGenes[i + 1]) in pairs:
      runs[-1].append(tempGenes[i + 1])
      continue
    runs.append([tempGenes[i + 1]])

  # find a reordering of runs that is better than the parent
  initialFitness = fnGetFitness(parentGenes)
  count = random.randint(2, 20)
  runIndexes = range(len(runs))
  while count > 0:
    count -= 1
    for i in runIndexes:
      if len(runs[i]) == 1:
        continue
      if random.randint(0, len(runs)) == 0:
        runs[i] = [n for n in reversed(runs[i])]

    indexA, indexB = random.sample(runIndexes, 2)
    runs[indexA], runs[indexB] = runs[indexB], runs[indexA]
    childGenes = list(chain.from_iterable(runs))
    if fnGetFitness(childGenes) > initialFitness:
      return childGenes
  return childGenes

# calculate cost (transition time) of genes
def get_time(productA, productB):
  tech_cost = tech_dict[productA['tech']][productB['tech']]
  gsm_cost = gsm_dict[productA['gsm']][productB['gsm']]
  tot_time = tech_cost + gsm_cost
  return tot_time

# define the fitness of the genes
class Fitness:
  def __init__(self, totalTime):
    self.TotalTime = totalTime

  def __gt__(self, other):
    return self.TotalTime < other.TotalTime

  def __str__(self):
    return "{:0.2f}".format(self.TotalTime)

def get_fitness(genes, orders):
  fitness = 0
  for i in range(len(genes)-1):
    start = orders[genes[i]]
    end = orders[genes[i+1]]
    fitness += get_time(start, end)
  return Fitness(round(fitness, 2))

# define the display function
def display(candidate, startTime):
    timeDiff = datetime.datetime.now() - startTime
    print("{}\t{}\t{}\t{}".format(
        ' '.join(map(str, candidate.Genes)),
        candidate.Fitness,
        candidate.Strategy.name,
        timeDiff))

# define how we mutate our genes
def mutate(genes, fnGetFitness):
  count = random.randint(2, len(genes))
  initialFitness = fnGetFitness(genes)
  while count > 0:
    count -= 1
    indexA, indexB = random.sample(range(len(genes)), 2)
    genes[indexA], genes[indexB] = genes[indexB], genes[indexA]
    fitness = fnGetFitness(genes)
    if fitness > initialFitness:
      return

geneset = [i for i in orders.keys()]

def fnCreate():
  return random.sample(geneset, len(geneset))

def fnDisplay(candidate):
  wheel = []
  for idx in candidate.Genes:
    wheel.append(orders[idx])
  wheel = pd.DataFrame(wheel)
  wheel.to_csv('results.csv',index=True)
  display(candidate, startTime)

def fnGetFitness(genes):
  return get_fitness(genes, orders)

def fnMutate(genes):
  mutate(genes, fnGetFitness)

def fnCrossover(parent, donor):
  return crossover(parent, donor, fnGetFitness)

optimalFitness = fnGetFitness(geneset)
print(optimalFitness)
startTime = datetime.datetime.now()
best = get_best(fnGetFitness,
                None,
                optimalFitness,
                None,
                fnDisplay,
                fnMutate,
                fnCreate,
                maxAge=500,
                poolSize=25,
                crossover=fnCrossover)

wheel = []
for idx in best.Genes:
  wheel.append(orders[idx])
wheel = pd.DataFrame(wheel)
wheel.to_csv('results.csv',index=True)
