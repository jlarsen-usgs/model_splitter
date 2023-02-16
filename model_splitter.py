import numpy as np
import flopy
from flopy.mf6.mfbase import PackageContainer
import matplotlib.pyplot as plt
import inspect


class Mf6Splitter(object):
    """
    A class for splitting a single model into a multi-model simulation

    Parameters
    ----------
    """
    def __init__(self, sim, modelname=None):
        self._sim = sim
        self._new_sim = None
        self._model = self._sim.get_model(modelname)
        if modelname is None:
            self._modelname = self._model.name
        self._model_type = self._model.model_type
        self._modelgrid = self._model.modelgrid
        self._node_map = {}
        self._new_connections = None
        self._new_ncpl = None
        self._grid_info = None
        self._exchange_metadata = None
        self._connection = None
        self._uconnection = None
        self._usg_metadata = None
        self._connection_ivert = None
        self._model_dict = None
        self._ivert_vert_remap = None

    @property
    def new_simulation(self):
        return self._new_sim

    def _remap_nodes(self, array):
        """
        Method to remap existing nodes to new models

        Parameters
        ----------
        array : numpy array
            numpy array of dimension ncpl (nnodes for DISU models)

        """
        array = np.ravel(array)

        if self._modelgrid.iverts is None:
            self._map_iac_ja_connections()
        else:
             self._map_connections()

        grid_info = {}
        if self._modelgrid.grid_type == "structured":
            a = array.reshape(self._modelgrid.nrow, self._modelgrid.ncol)
            for m in np.unique(a):
                cells = np.where(a == m)
                rmin, rmax = np.min(cells[0]), np.max(cells[0])
                cmin, cmax = np.min(cells[1]), np.max(cells[1])
                cells = list(zip(cells[0], cells[1]))
                # get new nrow and ncol information
                nrow = (rmax - rmin) + 1
                ncol = (cmax - cmin) + 1
                mapping = np.ones((nrow, ncol), dtype=int) * -1
                for ix, oix in enumerate(range(rmin, rmax + 1)):
                    for jx, jix in enumerate(range(cmin, cmax + 1)):
                        if (oix, jix) in cells:
                            onode = self._modelgrid.get_node((0, oix, jix))[0]
                            mapping[ix, jx] = onode
                grid_info[m] = [(nrow, ncol), (rmin, rmax), (cmin, cmax),
                                np.ravel(mapping)]
        else:
            for m in np.unique(array):
                cells = np.where(array == m)
                grid_info[m] = [(len(cells[0]),), None, None, None]

        new_ncpl = {}
        for m in np.unique(array):
            new_ncpl[m] = 1
            for i in grid_info[m][0]:
                new_ncpl[m] *= i

        for mdl in np.unique(array):
            mnodes = np.where(array == mdl)[0]
            mg_info = grid_info[mdl]
            for nnode, onode in enumerate(mnodes):
                if mg_info[-1] is not None:
                    nnode = np.where(mg_info[-1] == onode)[0][0]
                self._node_map[onode] = (mdl, nnode)

        new_connections = {i: {"internal": {},
                               "external": {}
                               } for i in np.unique(array)}
        exchange_meta = {i: {} for i in np.unique(array)}
        usg_meta = {i: {} for i in np.unique(array)}
        for node, conn in self._connection.items():
            mdl, nnode = self._node_map[node]
            for ix, cnode in enumerate(conn):
                cmdl, cnnode = self._node_map[cnode]
                if cmdl == mdl:
                    if nnode in new_connections[mdl]["internal"]:
                        new_connections[mdl]["internal"][nnode].append(cnnode)
                        if self._connection_ivert is None:
                            usg_meta[mdl][nnode]["ihc"].append(
                                int(self._uconnection[node]["ihc"][ix + 1])
                            )
                            usg_meta[mdl][nnode]["cl12"].append(
                                self._uconnection[node]["cl12"][ix + 1]
                            )
                            usg_meta[mdl][nnode]["hwva"].append(
                                self._uconnection[node]["hwva"][ix + 1]
                            )

                    else:
                        new_connections[mdl]["internal"][nnode] = [cnnode]
                        if self._connection_ivert is None:
                            usg_meta[mdl][nnode] = {
                                "ihc": [self._uconnection[node]["ihc"][0],
                                        self._uconnection[node]["ihc"][ix + 1]],
                                "cl12": [self._uconnection[node]["cl12"][0],
                                         self._uconnection[node]["cl12"][ix + 1]],
                                "hwva": [self._uconnection[node]["hwva"][0],
                                         self._uconnection[node]["hwva"][ix + 1]]
                            }

                else:
                    if nnode in new_connections[mdl]["external"]:
                        new_connections[mdl]["external"][nnode].append(
                            (cmdl, cnnode)
                        )
                        if self._connection_ivert is not None:
                            exchange_meta[mdl][nnode][cnnode] = \
                                [node, cnode, self._connection_ivert[node][ix]]
                        else:
                            exchange_meta[mdl][nnode][cnnode] =  [
                                node, cnode,
                                self._uconnection[node]["ihc"][ix + 1],
                                self._uconnection[node]["cl12"][ix + 1],
                                self._uconnection[node]["hwva"][ix + 1]
                            ]
                    else:
                        new_connections[mdl]["external"][nnode] = [
                            (cmdl, cnnode)]
                        if self._connection_ivert is not None:
                            exchange_meta[mdl][nnode] = \
                                {cnnode: [node, cnode, self._connection_ivert[node][ix]]}
                        else:
                            exchange_meta[mdl][nnode] = \
                                {cnnode: [
                                    node, cnode,
                                    self._uconnection[node]["ihc"][ix + 1],
                                    self._uconnection[node]["cl12"][ix + 1],
                                    self._uconnection[node]["hwva"][ix + 1]
                                ]
                                }

        # todo: remap iverts and verts
        if self._modelgrid.grid_type == "vertex":
            self._map_verts_iverts(array)

        self._new_connections = new_connections
        self._new_ncpl = new_ncpl
        self._grid_info = grid_info
        self._exchange_metadata = exchange_meta
        self._usg_metadata = usg_meta

    def _map_iac_ja_connections(self):
        """
        Method to map connections in unstructured grids when no
        vertex information has been supplied

        """
        conn = {}
        uconn = {}
        # todo: put in PR removing the MFArray component in UnstructuredGrid
        iac = self._model.disu.iac.array
        ja = self._model.disu.ja.array
        cl12 = self._model.disu.cl12.array
        ihc = self._model.disu.ihc.array
        hwva = self._model.disu.hwva.array
        idx0 = 0
        for ia in iac:
            idx1 = idx0 + ia
            cn = ja[idx0 + 1: idx1]
            conn[ja[idx0]] = cn
            uconn[ja[idx0]] = {
                "cl12": cl12[idx0: idx1],
                "ihc": ihc[idx0: idx1],
                "hwva": hwva[idx0: idx1]
            }
            idx0 = idx1

        self._connection = conn
        self._uconnection = uconn

    def _map_connections(self):
        iverts = self._modelgrid.iverts
        iverts = self._irregular_shape_patch(iverts)  # todo: adapt flopy's plotutil
        iv_r = iverts[:, ::-1]

        conn = {}
        connivert = {}
        # todo: try updating neighbors with this change for speed improvements
        for node, iv in enumerate(iverts):
            cells = []
            vix = []
            for i in range(1, len(iv)):
                i0 = i - 1
                if iv[i] == iv[i0]:
                    continue
                for ii in range(1, len(iv)):
                    ii0 = ii - 1
                    idxn = np.where(
                        (iv_r[:, ii0] == iv[i0]) & (iv_r[:, ii] == iv[i]))
                    if len(idxn[0]) > 0:
                        for n in idxn[0]:
                            if n != node:
                                cells.append(n)
                                vix.append((iv[i0], iv[i]))

            conn[node] = np.array(cells)
            connivert[node] = vix
        self._connection = conn
        self._connection_ivert = connivert

    def _map_verts_iverts(self, array):
        """

        Parameters
        ----------
        array :

        Returns
        -------

        """
        iverts = self._modelgrid.iverts
        verts = self._modelgrid.verts

        ivlut = {mkey: {} for mkey in np.unique(array)}
        for mkey in np.unique(array):
            new_iv = 0
            new_iverts = []
            new_verts = []
            tmp_vert_dict = {}
            for node, ivert in enumerate(iverts):
                tiverts = []
                mk, nnode = self._node_map[node]
                if mk == mkey:
                    for iv in ivert:
                        vert = tuple(verts[iv].tolist())
                        if vert in tmp_vert_dict:
                            tiverts.append(tmp_vert_dict[vert])
                        else:
                            tiverts.append(new_iv)
                            new_verts.append([new_iv] + list(vert))
                            tmp_vert_dict[vert] = new_iv
                            new_iv += 1

                    new_iverts.append(tiverts)

            ivlut[mkey]["iverts"] = new_iverts
            ivlut[mkey]["vertices"] = new_verts

        self._ivert_vert_remap = ivlut

    def _irregular_shape_patch(self, xverts, yverts=None):
        """
        DEPRECATED: remove and adapt plot_util's in final version

        Patch for vertex cross section plotting when
        we have an irregular shape type throughout the
        model grid or multiple shape types.

        Parameters
        ----------
        xverts : list
            xvertices
        yverts : list
            yvertices

        Returns
        -------
            xverts, yverts as np.ndarray

        """
        max_verts = 0

        for xv in xverts:
            if len(xv) > max_verts:
                max_verts = len(xv)

        if yverts is not None:
            for yv in yverts:
                if len(yv) > max_verts:
                    max_verts = len(yv)

        adj_xverts = []
        for xv in xverts:
            if len(xv) < max_verts:
                xv = list(xv)
                n = max_verts - len(xv)
                adj_xverts.append(xv + [xv[-1]] * n)
            else:
                adj_xverts.append(xv)

        if yverts is not None:
            adj_yverts = []
            for yv in yverts:
                if len(yv) < max_verts:
                    yv = list(yv)
                    n = max_verts - len(yv)
                    adj_yverts.append(yv + [yv[-1]] * n)
                else:
                    adj_yverts.append(yv)

            xverts = np.array(adj_xverts)
            yverts = np.array(adj_yverts)

            return xverts, yverts

        txverts = np.array(adj_xverts)
        xverts = np.zeros((txverts.shape[0], txverts.shape[1] + 1), dtype=int)
        xverts[:, 0:-1] = txverts
        xverts[:, -1] = xverts[:, 0]
        return xverts

    def _create_sln_tdis(self):
        """
        Method to create and add new TDIS and Solution Group objects
        from an existing Simulation

        Parameters
        ----------
        sim : MFSimulation object
            Simulation object that has a model that's being split
        new_sim : MFSimulation object
            New simulation object that will hold the split models

        Returns
        -------
            new_sim : MFSimulation object
        """
        for pak in self._sim.sim_package_list:
            pak_cls = PackageContainer.package_factory(pak.package_abbr,
                                                       "")
            signature = inspect.signature(pak_cls)
            d = {"simulation": self._new_sim, "loading_package": False}
            for key, value in signature.parameters.items():
                if key in ("simulation", "loading_package", "pname", "kwargs"):
                    continue
                elif key == "ats_perioddata":
                    # todo: figure out what needs to be done here. Child packages
                    #   seem like they are going to be a bit of an issue
                    continue
                else:
                    data = getattr(pak, key)
                    if hasattr(data, "array"):
                        d[key] = data.array
                    else:
                        d[key] = data

            new_pak = pak_cls(**d)

    def _remap_cell2d(self, item, cell2d, mapped_data):
        """
        Method to remap vertex grid cell2d

        Parameters
        ----------
        item : str
            parameter name string
        cell2d : MFList
            MFList object
        mapped_data : dict
            dictionary of remapped package data

        Returns
        -------
            dict
        """
        cell2d = cell2d.array

        for mkey in self._model_dict.keys():
            idx = []
            for node, (nmkey, nnode) in self._node_map.items():
                if nmkey == mkey:
                    idx.append(node)

            recarray = cell2d[idx]
            recarray["icell2d"] = range(len(recarray))
            iverts = self._irregular_shape_patch(self._ivert_vert_remap[mkey]["iverts"]).T
            for ix, ivert_col in enumerate(iverts[:-1]):
                recarray[f"icvert_{ix}"] = ivert_col

            mapped_data[mkey][item] = recarray

        return mapped_data

    def _remap_filerecords(self, item, value, mapped_data):
        """
        Method to create new file record names and map them to their
        associated models

        Parameters
        ----------
        item : str
            parameter name string
        value : MFList
            MFList object
        mapped_data : dict
            dictionary of remapped package data

        Returns
        -------
            dict
        """
        if item in (
                "budget_filerecord",
                "head_filerecord",
                "budgetcsv_filerecord"
        ):
            value = value.array
            if value is None:
                pass
            else:
                value = value[0][0]
                for mdl in mapped_data.keys():
                    new_val = value.split(".")
                    new_val = f"{new_val[0]}_{mdl}.{new_val[1]}"
                    mapped_data[mdl][item] = new_val
        return mapped_data

    def _remap_disu(self, pak, mapped_data):
        """
        Method to remap DISU inputs to new grids

        pak :
        mapped_data :

        Returns
        -------
        dict
        """
        for mkey, metadata in self._usg_metadata.items():
            iac, ja, ihc, cl12, hwva = [], [], [], [], []
            for node, params in metadata.items():
                conns = [node] + self._new_connections[mkey]["internal"][node]
                iac.append(len(conns))
                ja.extend(conns)
                ihc.extend(params["ihc"])
                cl12.extend(params["cl12"])
                hwva.extend(params["hwva"])

            print(np.sum(iac))
            print(len(ja))
            assert np.sum(iac) == len(ja)

            mapped_data[mkey]["nja"] = len(ja)
            mapped_data[mkey]["iac"] = iac
            mapped_data[mkey]["ja"] = ja
            mapped_data[mkey]["ihc"] = ihc
            mapped_data[mkey]["cl12"] = cl12
            mapped_data[mkey]["hwva"] = hwva

        return mapped_data

    def _remap_array(self, item, mfarray, mapped_data):
        """
        Method to remap array nodes to each model

        Parameters
        ----------
        item : str
            variable name
        value : MFArray
            MFArray object
        mapped_data : dict
            dictionary of remapped package data

        Returns
        -------
            dict
        """
        if mfarray.array is None:
            return mapped_data

        if not hasattr(mfarray, "size"):
            mfarray = mfarray.array
        # todo: compare array against ncpl to determine if it's 2d or 3d
        nlay = 1
        if isinstance(self._modelgrid.ncpl, (list, np.ndarray)):
            ncpl = self._modelgrid.nnodes
        else:
            ncpl = self._modelgrid.ncpl

        if mfarray.size == self._modelgrid.size:
            nlay = int(mfarray.size / ncpl)

        original_arr = mfarray.ravel()
        dtype = original_arr.dtype
        for layer in range(nlay):
            for node, (mkey, new_node) in self._node_map.items():
                node = node + (layer * ncpl)
                new_ncpl = self._new_ncpl[mkey]
                new_node = new_node + (layer * new_ncpl)
                value = original_arr[node]

                if item not in mapped_data[mkey]:
                    mapped_data[mkey][item] = np.zeros(new_ncpl * nlay, dtype=dtype)

                mapped_data[mkey][item][new_node] = value

        return mapped_data

    def _remap_mflist(self, item, mflist, mapped_data):
        """
        Method to remap mflist data to each model

        Parameters
        ----------
        item : str
            variable name
        value : MFList
            MFList object
        mapped_data : dict
            dictionary of remapped package data

        Returns
        -------
            dict
        """
        if hasattr(mflist, "array"):
            if mflist.array is None:
                return mapped_data
            recarray = mflist.array
        else:
            recarray = mflist

        if "cellid" not in recarray.dtype.names:
            for model in self._model_dict.keys():
                mapped_data[model][item] = recarray.copy()
        else:
            cellids = mflist.cellid
            if self._modelgrid.grid_type in ("structured", "vertex"):
                lay_num = np.array([i[0] for i in cellids])
                if self._modelgrid.grid_type == "structured":
                    nodes = self._modelgrid.get_node(cellids.tolist())
                    nodes -= (lay_num * self._modelgrid.ncpl)
                else:
                    nodes = np.array([i[1] for i in cellids])

            else:
                nodes = np.array([i[0] for i in cellids])

            new_model = np.zeros(cellids.shape)
            new_node = np.zeros(cellids.shape)
            for ix, node in enumerate(nodes):
                nm, nn = self._node_map[node]
                new_model[ix] = nm
                new_node[ix] = nn

            for mkey, model in self._model_dict.items():
                idx = np.where(new_model == mkey)[0]
                if len(idx) == 0:
                    new_recarray = None
                else:
                    model_node = new_node[idx].astype(int)
                    if self._modelgrid.grid_type == "structured":
                        model_node += (lay_num[idx] * model.modelgrid.ncpl)
                        new_cellids = model.modelgrid.get_lrc(
                            model_node.astype(int)
                        )
                    elif self._modelgrid.grid_type == "vertex":
                        new_cellids = [
                            tuple(cid) for cid in zip(lay_num[idx], model_node)
                        ]

                    else:
                        new_cellids = [(i,) for i in model_node]

                    new_recarray = recarray[idx]
                    new_recarray["cellid"] = new_cellids

                mapped_data[mkey][item] = new_recarray

        return mapped_data

    def _remap_transient_list(self, item, mftransientlist, mapped_data):
        """
        Method to remap transient list data to each model

        Parameters
        ----------
        item : str
            parameter name
        mftransientlist : MFTransientList
            MFTransientList object
        mapped_data : dict
            dictionary of remapped package data

        Returns
        -------
            dict
        """
        d0 = {mkey: {} for mkey in self._model_dict.keys()}
        for per, recarray in mftransientlist.data.items():
            d = self._remap_mflist(item, recarray, mapped_data)
            for mkey in self._model_dict.keys():
                if mapped_data[mkey][item] is None:
                    continue
                d0[mkey][per] = mapped_data[mkey][item]

        for mkey in self._model_dict.keys():
            mapped_data[mkey][item] = d0[mkey]

        return mapped_data

    def _remap_package(self, package):
        """
        Method to remap package data to new packages in each model

        Parameters
        ----------
        package : flopy.mf6.Package
            Package object

        Returns
        -------
            dict
        """
        # todo: MVR package! needs to be handled differently!!!!
        #   for MVR's that cross model boundaries a seperate dict will need
        #   to be compiled and then a MVR will need to be built with the
        #   exchange model

        # todo: child packages??? This is an issue that still needs solving.

        # todo: need a DISU trap for ja, ia, hvwa
        mapped_data = {mkey: {} for mkey in self._model_dict.keys()}
        if isinstance(
                package,
                (flopy.mf6.modflow.ModflowGwfdis,
                 flopy.mf6.modflow.ModflowGwfdisu,
                 flopy.mf6.modflow.ModflowGwtdis)
        ):
            for item, value in package.__dict__.items():
                if item in ('delr', "delc"):
                    for mkey, d in self._grid_info.items():
                        if item == "delr":
                            i0, i1 = d[2]
                        else:
                            i0, i1 = d[1]

                        mapped_data[mkey][item] = value.array[i0:i1 + 1]

                elif item in ("nrow", "ncol"):
                    for mkey, d in self._grid_info.items():
                        if item == "nrow":
                            i0, i1 = d[1]
                        else:
                            i0, i1 = d[2]

                        mapped_data[mkey][item] = (i1 - i0) + 1

                elif item == "nlay":
                    continue

                elif item == "nodes":
                    for mkey in self._model_dict.keys():
                        mapped_data[mkey][item] = self._grid_info[mkey][0][0]

                elif item == "iac":
                    mapped_data = self._remap_disu(package, mapped_data)
                    break

                elif isinstance(value, flopy.mf6.data.mfdataarray.MFArray):
                    mapped_data = self._remap_array(item, value, mapped_data)

        else:
            for item, value in package.__dict__.items():
                if item.startswith("_"):
                    continue

                elif item == "nvert":
                    continue

                elif item in ("ncpl", "nodes"):
                    for mkey in self._model_dict.keys():
                        mapped_data[mkey][item] = self._grid_info[mkey][0][0]

                elif item.endswith("_filerecord"):
                    mapped_data = self._remap_filerecords(item, value, mapped_data)

                elif item in ('vertices', "cell2d"):
                    if value.array is not None:
                        if item == "cell2d":
                            mapped_data = self._remap_cell2d(item, value, mapped_data)
                        else:
                            for mkey in self._model_dict.keys():
                                mapped_data[mkey][item] = self._ivert_vert_remap[mkey][item]
                                mapped_data[mkey]["nvert"] = len(self._ivert_vert_remap[mkey][item])

                elif isinstance(value, flopy.mf6.data.mfdataarray.MFArray):
                    mapped_data = self._remap_array(item, value, mapped_data)

                elif isinstance(value, flopy.mf6.data.mfdatalist.MFTransientList):
                    mapped_data = self._remap_transient_list(item, value, mapped_data)

                elif isinstance(value, flopy.mf6.data.mfdatalist.MFList):
                    mapped_data = self._remap_mflist(item, value, mapped_data)

                elif isinstance(value, flopy.mf6.data.mfdatascalar.MFScalar):
                    for mkey in self._model_dict.keys():
                        mapped_data[mkey][item] = value.data
                else:
                    print('break')

        if "options" in package.blocks:
            for item, value in package.blocks["options"].datasets.items():
                if item.endswith("_filerecord"):
                    mapped_data = self._remap_filerecords(item, value, mapped_data)
                    continue

                for mkey in self._model_dict.keys():
                    if isinstance(value, flopy.mf6.data.mfdatascalar.MFScalar):
                        mapped_data[mkey][item] = value.data
                    elif isinstance(value, flopy.mf6.data.mfdatalist.MFList):
                        mapped_data[mkey][item] = value.array

        pak_cls = PackageContainer.package_factory(package.package_type,
                                                   package.parent.model_type)
        paks = {}
        for mdl, data in mapped_data.items():
            paks[mdl] = pak_cls(self._model_dict[mdl], **data)

        return paks

    def _create_exchanges(self):
        """
        Method to create exchange packages for fluxes between models

        Returns
        -------
            dict
        """
        # todo: handle MVR here?
        #   or build it somewhere else and pass in the MVR perioddata?
        d = {}
        built = []
        nmodels = len(self._model_dict)
        if self._modelgrid.grid_type == "unstructured":
            # use existing connection information
            aux = False
            for m0, model in self._model_dict.items():
                exg_nodes = self._new_connections[m0]["external"]
                for m1 in range(nmodels):
                    if m1 in built:
                        continue
                    if m1 == m0:
                        continue
                    exchange_data = []
                    for node0, exg_list in exg_nodes.items():
                        for exg in exg_list:
                            if exg[0] != m1:
                                continue
                            node1 = exg[-1]
                            exg_meta0 = self._exchange_metadata[m0][node0][node1]
                            exg_meta1 = self._exchange_metadata[m1][node1][node0]
                            rec = ((node0,), (node1,), 1, exg_meta0[3], exg_meta1[3], exg_meta0[-1])
                            exchange_data.append(rec)

                    if exchange_data:
                        mname0 = self._model_dict[m0].name
                        mname1 = self._model_dict[m1].name
                        exchg = flopy.mf6.modflow.ModflowGwfgwf(
                            self._new_sim,
                            exgtype="GWF6-GWF6",
                            # todo open an issue about this! This should be explicitly set when running create_package.py!
                            exgmnamea=mname0,
                            exgmnameb=mname1,
                            nexg=len(exchange_data),
                            exchangedata=exchange_data
                        )
                        d[f"{mname0}_{mname1}"] = exchg

                built.append(m0)

            for _, model in self._model_dict.items():
                # turn off save_specific_discharge if it's on
                model.npf.save_specific_discharge = None

        else:
            xc = self._modelgrid.xcellcenters.ravel()
            yc = self._modelgrid.ycellcenters.ravel()
            verts = self._modelgrid.verts
            for m0, model in self._model_dict.items():
                exg_nodes = self._new_connections[m0]["external"]
                for m1 in range(nmodels):
                    if m1 in built:
                        continue
                    if m1 == m0:
                        continue
                    modelgrid0 = model.modelgrid
                    modelgrid1 = self._model_dict[m1].modelgrid
                    ncpl0 = modelgrid0.ncpl
                    ncpl1 = modelgrid1.ncpl
                    exchange_data = []
                    for node0, exg_list in exg_nodes.items():
                        for exg in exg_list:
                            if exg[0] != m1:
                                continue

                            node1 = exg[1]
                            for layer in range(self._modelgrid.nlay):
                                if self._modelgrid.grid_type == "structured":
                                    tmpnode0 = node0 + (ncpl0 * layer)
                                    tmpnode1 = node1 + (ncpl1 * layer)
                                    cellidm0 = modelgrid0.get_lrc([tmpnode0])[0]
                                    cellidm1 = modelgrid1.get_lrc([tmpnode1])[0]
                                elif self._modelgrid.grid_type == "vertex":
                                    cellidm0 = (layer, node0)
                                    cellidm1 = (layer, node1)
                                else:
                                    cellidm0 = node0
                                    cellidm1 = node1

                                if modelgrid0.idomain[cellidm0] == 0:
                                    continue
                                if modelgrid1.idomain[cellidm1] == 0:
                                    continue
                                # calculate CL1, CL2 from exchange metadata
                                meta = self._exchange_metadata[m0][node0][node1]
                                ivrt = meta[2]
                                x1 = xc[meta[0]]
                                y1 = yc[meta[0]]
                                x2 = xc[meta[1]]
                                y2 = yc[meta[1]]
                                x3, y3 = verts[ivrt[0]]
                                x4, y4 = verts[ivrt[1]]

                                numa = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
                                denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
                                ua = numa / denom
                                x = x1 + ua * (x2 - x1)
                                y = y1 + ua * (y2 - y1)

                                cl0 = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
                                cl1 = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
                                hwva = np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)

                                # calculate angledegx and cdist
                                angledegx = np.arctan2([x2 - x1], [y2 - y1, ])[0] * (180 / np.pi)
                                if angledegx < 0:
                                    angledegx = 360 + angledegx

                                cdist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                                rec = [cellidm0, cellidm1, 1, cl0, cl1, hwva, angledegx, cdist]
                                exchange_data.append(rec)

                    if exchange_data:
                        mname0 = self._model_dict[m0].name
                        mname1 = self._model_dict[m1].name
                        exchg = flopy.mf6.modflow.ModflowGwfgwf(
                            self._new_sim,
                            exgtype="GWF6-GWF6", # todo open an issue about this! This should be explicitly set when running create_package.py!
                            exgmnamea=mname0,
                            exgmnameb=mname1,
                            auxiliary=["ANGLDEGX", "CDIST"],
                            nexg=len(exchange_data),
                            exchangedata=exchange_data
                        )
                        d[f"{mname0}_{mname1}"] = exchg

                built.append(m0)

        return d

    def split_model(self, array):
        """

        Parameters
        ----------
        array :

        Returns
        -------
            MFSimulation object
        """
        self._remap_nodes(array)

        self._new_sim = flopy.mf6.MFSimulation()
        self._create_sln_tdis()

        self._model_dict = {}
        for mkey in self._new_ncpl.keys():
            mdl_cls = PackageContainer.model_factory(self._model_type)
            self._model_dict[mkey] = mdl_cls(
                self._new_sim, modelname=f"{self._modelname}_{mkey}"
            )

        for package in self._model.packagelist:
            paks = self._remap_package(package)

        epaks = self._create_exchanges()

        return self._new_sim


# todo: development notes:
#   Need to set up advanced packages starting with UZF, MAW, and LAK
#   Then set up MVR
#   Finally set up SFR and mover upstream model flow to downstream model flow
#   Then set up checks for model splitting
#       (ex. doesnt parallel a fault, doesnt cut through a lake)
#   Finally deal with subpackages...