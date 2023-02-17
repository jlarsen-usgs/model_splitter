import os
import numpy as np
import matplotlib.pyplot as plt
import flopy
from model_splitter import Mf6Splitter


ws = os.path.abspath(os.path.dirname(__file__))
sim_ws = os.path.join(ws, "examples", "data", "test001e_UZF_3lay")
sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws)
sim.run_simulation()

gwf = sim.get_model()
modelgrid = gwf.modelgrid
nrow, ncol = modelgrid.shape[1:]

array = np.zeros((nrow, ncol), dtype=int)
array[:, 5:] = 1

mfsplit = Mf6Splitter(sim)
new_sim = mfsplit.split_model(array)

new_sim.set_sim_path(os.path.join(ws, 'examples', 'output', 'UZF_3lay_split'))
new_sim.write_simulation()
new_sim.run_simulation()