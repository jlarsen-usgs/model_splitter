import os
import numpy as np
import matplotlib.pyplot as plt
import flopy
from model_splitter import Mf6Splitter


ws = os.path.abspath(os.path.dirname(__file__))
sim_ws = os.path.join(ws, "examples", "data", "disu_model")
sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws)

gwf = sim.get_model()
modelgrid = gwf.modelgrid
ncpl = modelgrid.nnodes

array = np.zeros((ncpl,), dtype=int)
array[65:] = 1

mfsplit = Mf6Splitter(sim)
new_sim = mfsplit.split_model(array)

new_sim.set_sim_path("disu_split")
new_sim.write_simulation()
new_sim.run_simulation()