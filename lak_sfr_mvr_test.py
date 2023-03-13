import os
import numpy as np
import matplotlib.pyplot as plt
import flopy
from model_splitter import Mf6Splitter


sim_ws = os.path.join(".", "examples", "data", "test045_lake2tr")
sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws)
ml = sim.get_model()

lak = ml.lak
array = np.zeros((ml.modelgrid.nrow, ml.modelgrid.ncol), dtype=int)
array[0:14, :] = 1

pmv = flopy.plot.PlotMapView(model=ml)
pmv.plot_grid()
pmv.plot_inactive()
pmv.plot_array(array)
pmv.plot_bc(package=lak)
pmv.plot_bc("SFR")
plt.show()

# sim.set_sim_path("temp")
# sim.write_simulation()
# sim.run_simulation()

mfsplit = Mf6Splitter(sim)
new_sim = mfsplit.split_model(array)

ml = new_sim.get_model("lakeex2a_0")

pmv = flopy.plot.PlotMapView(model=ml)
pmv.plot_bc("LAK")
pmv.plot_bc("SFR")
pmv.plot_grid()
plt.show()

new_sim.set_sim_path("temp")
new_sim.write_simulation()
new_sim.run_simulation()