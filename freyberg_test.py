import os
import flopy
import matplotlib.pyplot as plt
import numpy as np
from model_splitter import Mf6Splitter


ws = os.path.abspath(os.path.dirname(__file__))
ws = os.path.join(ws, "..", "flopy-dev")
sim_ws = os.path.join(ws, "examples", "data", "mf6-freyberg")
sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws)
sim.run_simulation()

gwf = sim.get_model()
modelgrid = gwf.modelgrid

# plot initial boundary condition
fig, ax = plt.subplots(figsize=(5, 7))
pmv = flopy.plot.PlotMapView(gwf, ax=ax)
heads = gwf.output.head().get_alldata()[-1]
heads = np.where(heads == 1e+30,
                 np.nan,
                 heads)
vmin = np.nanmin(heads)
vmax = np.nanmax(heads)
pc = pmv.plot_array(heads, vmin=vmin, vmax=vmax)
pmv.plot_bc("WEL")
pmv.plot_bc("RIV", color="c")
pmv.plot_bc("CHD")
pmv.plot_grid()
pmv.plot_ibound()
plt.colorbar(pc)
plt.show()


array = np.zeros((modelgrid.nrow, modelgrid.ncol), dtype=int)
ncol = 1
for row in range(modelgrid.nrow):
    if row != 0 and row % 2 == 0:
        ncol += 1
    array[row, ncol:] = 1

fig, ax = plt.subplots(figsize=(5, 7))
pmv = flopy.plot.PlotMapView(gwf, ax=ax)
pc = pmv.plot_array(array)
lc = pmv.plot_grid()
plt.colorbar(pc)
plt.show()

mfsplit = Mf6Splitter(sim)
new_sim = mfsplit.split_model(array)
new_sim.set_sim_path("dis_split")
new_sim.write_simulation()
new_sim.run_simulation()

ml0 = new_sim.get_model("freyberg_0")
ml1 = new_sim.get_model("freyberg_1")

heads0 = ml0.output.head().get_alldata()[-1]
heads1 = ml1.output.head().get_alldata()[-1]

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 7))
pmv = flopy.plot.PlotMapView(ml0, ax=ax0)
pmv.plot_array(heads0, vmin=vmin, vmax=vmax)
pmv.plot_ibound()
pmv.plot_grid()
pmv.plot_bc("WEL")
pmv.plot_bc("RIV", color="c")
pmv.plot_bc("CHD")
ax0.set_title("Model 0")

pmv = flopy.plot.PlotMapView(ml1, ax=ax1)
pc = pmv.plot_array(heads1, vmin=vmin, vmax=vmax)
pmv.plot_ibound()
pmv.plot_bc("WEL")
pmv.plot_bc("RIV", color="c")
pmv.plot_grid()
ax1.set_title("Model 1")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(pc, cax=cbar_ax, label="Hydraulic heads")
plt.show()