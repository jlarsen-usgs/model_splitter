import flopy
from model_splitter import Mf6Splitter
import numpy as np
import os


ws = os.path.join("examples", "data", "test036_twrihfb")

sim = flopy.mf6.MFSimulation.load(sim_ws=ws)
ml = sim.get_model()

modelgrid = ml.modelgrid

array = np.zeros((modelgrid.nrow, modelgrid.ncol), dtype=int)
array[8:, :] = 1

mfsplit = Mf6Splitter(sim)
new_sim = mfsplit.split_model(array)
new_sim.set_sim_path("temp2")
new_sim.write_simulation()
new_sim.run_simulation()
print('break')