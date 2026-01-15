"""Mesh channel knowledge (CK) map"""
import mitsuba as mi
import drjit as dr
from typing import List, Tuple
from scipy.constants import speed_of_light

from sionna.rt.scene import Scene
from .ck_map import CKMap
from sionna.rt.utils import WedgeGeometry, wedge_interior_angle

class MeshCKMap(CKMap):
    r"""
    Mesh CK Map (Extended metrics support)

    A mesh-based CK map is computed by a solver for a measurement surface
    defined by a mesh. Each triangle of the mesh serves as a bin.
    It supports advanced metrics: Path Gain, ToA, RMS-DS, DoA, DoD, LoS.

    :param scene: Scene for which the radio map is computed
    :param meas_surface: Mesh to be used as the measurement surface
    """
    def __init__(self, scene: Scene, meas_surface: mi.Mesh):

        super().__init__(scene)

        self._cells_count = meas_surface.face_count()
        self._meas_surface = meas_surface

        # -----------------------------------------------------------
        # Buffer Initialization
        # -----------------------------------------------------------
        
        # Shape: [num_tx, cells_count]
        map_shape = (self.num_tx, self._cells_count)
        
        # 1. Path Gain (Energy)
        self._pathgain_map = dr.zeros(mi.TensorXf, map_shape)

        # 2. RMS Delay Spread Accumulators
        self._delay_moment1 = dr.zeros(mi.TensorXf, map_shape) # Sum(P * t)
        self._delay_moment2 = dr.zeros(mi.TensorXf, map_shape) # Sum(P * t^2)
        self._rms_ds_map = dr.zeros(mi.TensorXf, map_shape)    # Result

        # 3. Time of Arrival (ToA)
        # Init with infinity for Min-Reduction
        self._toa_map = dr.full(mi.TensorXu, 0xFFFFFFFF, map_shape)

        # 4. LoS Flag
        self._is_los_map = dr.zeros(mi.TensorXi, map_shape)

        # 5. DoA and DoD Accumulators
        # Shape: [num_tx, cells_count, 3]
        vec_map_shape = (self.num_tx, self._cells_count, 3)
        self._doa_acc = dr.zeros(mi.TensorXf, vec_map_shape)
        self._dod_acc = dr.zeros(mi.TensorXf, vec_map_shape)
        self._mean_doa_map = dr.zeros(mi.TensorXf, vec_map_shape) # Result
        self._mean_dod_map = dr.zeros(mi.TensorXf, vec_map_shape) # Result

    # -----------------------------------------------------------
    # Properties Implementation
    # -----------------------------------------------------------

    @property
    def measurement_surface(self):
        r"""Mitsuba shape corresponding to the
        radio map measurement surface

        :type: :py:class:`mi.Mesh`
        """
        return self._meas_surface

    @property
    def cells_count(self):
        r"""Total number of cells in the radio map

        :type: :py:class:`int`
        """
        return self._cells_count

    @property
    def cell_centers(self):
        r"""Positions of the centers of the cells in the global coordinate
        system

        :type: :py:class:`mi.Point3f [cells_count, 3]`
        """
        mesh = self.measurement_surface
        ind = mesh.face_indices(dr.arange(mi.UInt, 0, mesh.face_count()))
        v = mesh.vertex_position(dr.ravel(ind))
        v_x = dr.reshape(mi.TensorXf, v.x, [mesh.face_count(), 3])
        v_y = dr.reshape(mi.TensorXf, v.y, [mesh.face_count(), 3])
        v_z = dr.reshape(mi.TensorXf, v.z, [mesh.face_count(), 3])
        c_x = dr.mean(v_x, axis=1)
        c_y = dr.mean(v_y, axis=1)
        c_z = dr.mean(v_z, axis=1)
        c = mi.Point3f(c_x, c_y, c_z)
        return c

    @property
    def path_gain(self):
        # pylint: disable=line-too-long
        r"""Path gains across the radio map from all transmitters [unitless, linear scale]

        :type: :py:class:`mi.TensorXf [num_tx, num_primitives]`
        """
        return self._pathgain_map

    @property
    def rms_ds(self):
        # pylint: disable=line-too-long
        r"""RMS delay spread across the radio map from all transmitters [s]

        :type: :py:class:`mi.TensorXf [num_tx, cells_per_dim_y, cells_per_dim_x]`
        """
        return self._rms_ds_map

    @property
    def toa(self):
        # pylint: disable=line-too-long
        r"""Time of arrival (ToA) across the radio map from all transmitters [s]

        :type: :py:class:`mi.TensorXf [num_tx, cells_per_dim_y, cells_per_dim_x]`
        """
        return self._toa_map

    @property
    def mean_doa(self):
        # pylint: disable=line-too-long
        r"""Mean direction of arrival (DoA) across the radio map from all transmitters [rad]

        :type: :py:class:`mi.TensorXf [num_tx, cells_per_dim_y, cells_per_dim_x, 3]`
        """
        return self._mean_doa_map

    @property
    def mean_dod(self):
        # pylint: disable=line-too-long
        r"""Mean direction of departure (DoD) across the radio map from all transmitters [rad]

        :type: :py:class:`mi.TensorXf [num_tx, cells_per_dim_y, cells_per_dim_x, 3]`
        """
        return self._mean_dod_map

    @property
    def is_los(self):
        # pylint: disable=line-too-long
        r"""Line-of-sight flag across the radio map from all transmitters

        :type: :py:class:`mi.TensorXi [num_tx, cells_per_dim_y, cells_per_dim_x]`
        """
        return self._is_los_map

    # -----------------------------------------------------------
    # Core Logic: add_paths
    # -----------------------------------------------------------

    def add_paths(
        self,
        e_fields: mi.Vector4f,
        array_w: List[mi.Float],
        si: mi.SurfaceInteraction3f,
        k_world: mi.Vector3f,
        tx_indices: mi.UInt,
        active: mi.Bool,
        diffracted_paths: bool,
        solid_angle: mi.Float | None = None,
        tx_positions: mi.Point3f | None = None,
        wedges: WedgeGeometry | None = None,
        diff_point: mi.Point3f | None = None,
        wedges_samples_cnt: mi.UInt | None = None,
        # New arguments
        total_dist: mi.Float | None = None,
        ray_start_dir: mi.Vector3f | None = None,
        is_los: mi.Bool | None = None):
        # pylint: disable=line-too-long
        r"""
        Adds the contribution of the paths that hit the measurement surface
        to the radio maps

        The radio maps are updated in place.

        :param e_fields: Electric fields as real-valued vectors of dimension 4
        :param array_w: Weighting used to model the effect of the transmitter
            array
        :param si: Informations about the interaction with the measurement
            surface
        :param k_world: Directions of propagation of the incident paths
        :param tx_indices: Indices of the transmitters from which the rays originate
        :param active: Flags indicating if the paths should be added to the radio map
        :param diffracted_paths: Flags indicating if the paths are diffracted
        :param solid_angle: Ray tubes solid angles [sr] for non-diffracted paths.
            Not required for diffracted paths.
        :param tx_positions: Positions of the transmitters
        :param wedges: Properties of the intersected wedges.
            Not required for non-diffracted paths.
        :param diff_point: Position of the diffraction point on the wedge.
            Not required for non-diffracted paths.
        :param wedges_samples_cnt: Number of samples on the wedge.
            Not required for non-diffracted paths.
        :param total_dist: Total distance traveled by the ray [m].
            Required for delay metrics.
        :param ray_start_dir: Direction of the ray at the start of the path
            [unit vector, 3D].
            Required for DoD computation.
        :param is_los: Line-of-sight flag.
            Required for delay metrics.
        """
        # Indices of the hit cells is the primitive ID
        tensor_ind = tx_indices * self.cells_count + si.prim_index

        # Contribution to the path loss map
        a = dr.zeros(mi.Vector4f, 1)
        for e_field, aw in zip(e_fields, array_w):
            a += aw @ e_field
        power_unweighted = dr.squared_norm(a)

        # Ray weight for non-diffracted paths
        if not diffracted_paths:
            cos_theta = dr.abs(dr.dot(si.n, k_world))
            w = solid_angle * dr.rcp(cos_theta)
        else:
            # Ray weight for diffracted paths
            tx_positions_ = dr.gather(mi.Point3f, tx_positions, tx_indices,
                                     active=active)
            w = self._diffraction_integration_weight(wedges, tx_positions_,
                                                     diff_point, k_world, si)
            w *= wedges.length * (dr.two_pi -
                                 wedge_interior_angle(wedges.n0, wedges.nn))
            w /= wedges_samples_cnt

        # Normalize by cell area
        meas_surface = self.measurement_surface
        prim_index = meas_surface.face_indices(si.prim_index, active=active)
        v0 = meas_surface.vertex_position(prim_index[0])
        v1 = meas_surface.vertex_position(prim_index[1])
        v2 = meas_surface.vertex_position(prim_index[2])
        v1 = v1 - v0
        v2 = v2 - v0
        v1_sq_norm = dr.squared_norm(v1)
        v2_sq_norm = dr.squared_norm(v2)
        cell_area = 0.5 * dr.sqrt(v1_sq_norm * v2_sq_norm
                                  - dr.square(dr.dot(v1, v2)))
        
        w *= dr.rcp(cell_area)

        # Final Power
        power = power_unweighted * w

        # --- A. Accumulate Path Gain ---
        dr.scatter_reduce(dr.ReduceOp.Add, self._pathgain_map.array, value=power,
                          index=tensor_ind, active=active)

        # --- B. Accumulate Delay Metrics ---
        if total_dist is not None:
            tau = total_dist / speed_of_light
            
            # Moment 1: P * tau
            dr.scatter_reduce(dr.ReduceOp.Add, self._delay_moment1.array, 
                              value=power * tau, index=tensor_ind, active=active)
            
            # Moment 2: P * tau^2
            dr.scatter_reduce(dr.ReduceOp.Add, self._delay_moment2.array, 
                              value=power * dr.sqr(tau), index=tensor_ind, active=active)

            # ToA (Min tau)
            # Only update ToA if the ray carries meaningful power
            valid_toa = active & (power > 1e-20)
            tau_ns = tau * 1e9
            tau_ns_uint = mi.UInt(dr.round(tau_ns))
            dr.scatter_reduce(dr.ReduceOp.Min, self._toa_map.array, 
                            value=tau_ns_uint, index=tensor_ind, active=valid_toa)

        # --- C. Accumulate LoS Flag ---
        if is_los is not None:
            los_val = dr.select(is_los, 1, 0)
            dr.scatter_reduce(dr.ReduceOp.Max, self._is_los_map.array,
                              value=los_val, index=tensor_ind, active=active)

        # --- D. Accumulate Direction Vectors ---
        # Tensor indices for vector maps: [num_tx * cells_count * 3]
        tensor_ind_vec = tensor_ind * 3

        # DoA
        doa_vec = -k_world
        weighted_doa = doa_vec * power
        
        dr.scatter_reduce(dr.ReduceOp.Add, self._doa_acc.array, value=weighted_doa.x,
                          index=tensor_ind_vec + 0, active=active)
        dr.scatter_reduce(dr.ReduceOp.Add, self._doa_acc.array, value=weighted_doa.y,
                          index=tensor_ind_vec + 1, active=active)
        dr.scatter_reduce(dr.ReduceOp.Add, self._doa_acc.array, value=weighted_doa.z,
                          index=tensor_ind_vec + 2, active=active)

        # DoD
        if ray_start_dir is not None:
            weighted_dod = ray_start_dir * power
            
            dr.scatter_reduce(dr.ReduceOp.Add, self._dod_acc.array, value=weighted_dod.x,
                              index=tensor_ind_vec + 0, active=active)
            dr.scatter_reduce(dr.ReduceOp.Add, self._dod_acc.array, value=weighted_dod.y,
                              index=tensor_ind_vec + 1, active=active)
            dr.scatter_reduce(dr.ReduceOp.Add, self._dod_acc.array, value=weighted_dod.z,
                              index=tensor_ind_vec + 2, active=active)

    def finalize(self):
        """Finalizes the computation of the radio map."""

        # Scale the maps
        wavelength = self._wavelength
        scaling = dr.square(wavelength * dr.rcp(4. * dr.pi))
        
        # Apply scaling to all power-dependent accumulators
        self._pathgain_map *= scaling
        self._delay_moment1 *= scaling
        self._delay_moment2 *= scaling
        self._doa_acc *= scaling
        self._dod_acc *= scaling

        # -------------------------------------------------------
        # 2. Finalize Derived Metrics
        # -------------------------------------------------------
        
        # Helper: Inverse Power (avoid div by zero)
        total_power = dr.maximum(self._pathgain_map, 1e-30)
        inv_power = dr.rcp(total_power)
        
        # Mask for valid cells (where we actually received energy)
        valid_mask = self._pathgain_map > 0

        # --- RMS Delay Spread ---
        mean_delay = self._delay_moment1 * inv_power
        mean_sq_delay = self._delay_moment2 * inv_power
        variance = mean_sq_delay - dr.sqr(mean_delay)
        variance = dr.maximum(variance, 0.0) # Numerical stability
        
        self._rms_ds_map = dr.safe_sqrt(variance)
        self._rms_ds_map = dr.select(valid_mask, self._rms_ds_map, 0.0)

        # --- Mean DoA / DoD ---
        # Need to multiply [Tx, Y, X, 3] by [Tx, Y, X] (inv_power)
        # FIX: Use dr.reshape instead of dr.unsqueeze
        current_shape = list(inv_power.shape)
        new_shape = current_shape + [1]
        
        inv_power_expanded = dr.reshape(mi.TensorXf, inv_power, new_shape)
        
        # Normalize accumulators by total power
        self._mean_doa_map = self._doa_acc * inv_power_expanded
        self._mean_dod_map = self._dod_acc * inv_power_expanded
        
        # Zero out invalid cells
        # Ensure mask is also reshaped and correct type for broadcasting
        valid_mask_expanded = dr.reshape(dr.mask_t(mi.TensorXf), valid_mask, new_shape)
        
        self._mean_doa_map = dr.select(valid_mask_expanded, self._mean_doa_map, 0.0)
        self._mean_dod_map = dr.select(valid_mask_expanded, self._mean_dod_map, 0.0)

        # --- ToA Cleanup & Conversion ---
        # 1. Get flat UInt array
        toa_uint_flat = self._toa_map.array
        
        # 2. Convert to Float seconds
        toa_float_flat = mi.Float(toa_uint_flat) * 1e-9
        
        # 3. Handle Sentinel (0xFFFFFFFF -> -1.0)
        # Use comparison on the uint array, apply result to float array
        sentinel = 0xFFFFFFFF
        toa_float_flat = dr.select(toa_uint_flat == sentinel, -1.0, toa_float_flat)
        
        # 4. Reconstruct TensorXf
        # We must explicitly provide the shape to recreate the Tensor
        self._toa_map = mi.TensorXf(toa_float_flat, self._toa_map.shape)

    def sample_positions(
        self,
        num_pos: int,
        metric: str = "path_gain",
        min_val_db: float | None = None,
        max_val_db: float | None = None,
        min_dist: float | None = None,
        max_dist: float | None = None,
        tx_association: bool = True,
        center_pos: bool = False,
        seed: int = 1
        ) -> Tuple[mi.TensorXf, mi.TensorXu]:
        # pylint: disable=line-too-long
        r"""Samples random user positions in a scene based on a radio map

        For a given radio map, ``num_pos`` random positions are sampled
        around each transmitter, such that the selected metric, e.g., SINR, is
        larger than ``min_val_db`` and/or smaller than ``max_val_db``.
        Similarly, ``min_dist`` and ``max_dist`` define the minimum and maximum
        distance of the random positions to the transmitter under consideration.
        By activating the flag ``tx_association``, only positions are sampled
        for which the selected metric is the highest across all transmitters.
        This is useful if one wants to ensure, e.g., that the sampled positions
        for each transmitter provide the highest SINR or RSS.

        Note that due to the quantization of the radio map into cells it is
        not guaranteed that all above parameters are exactly fulfilled for a
        returned position. This stems from the fact that every
        individual cell of the radio map describes the expected *average*
        behavior of the surface within this cell. For instance, it may happen
        that half of the selected cell is shadowed and, thus, no path to the
        transmitter exists but the average path gain is still larger than the
        given threshold. Please enable the flag ``center_pos`` to sample only
        positions from the cell centers.

        .. code-block:: Python

            import numpy as np
            import sionna
            from sionna.rt import load_scene, PlanarArray, Transmitter, \
                                  RadioMapSolver, Receiver, transform_mesh

            scene = load_scene(sionna.rt.scene.san_francisco)

            # Configure antenna array for all transmitters
            scene.tx_array = PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.7,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V")
            # Add a transmitters
            tx = Transmitter(name="tx",
                        position=[15.9,121.2,25],
                        orientation=[0,0,0],
                        display_radius=2.0)
            scene.add(tx)

            # Create a measurement surface by cloning the terrain
            # and elevating it by 1.5 meters
            measurement_surface = scene.objects["Terrain"].clone(as_mesh=True)
            transform_mesh(measurement_surface,
                        translation=[0,0,1.5])

            solver = RadioMapSolver()
            rm = solver(scene,
                        cell_size=(1., 1.),
                        measurement_surface=measurement_surface,
                        samples_per_tx=100000000)

            positions,_ = rm.sample_positions(num_pos=200, min_val_db=-100.,
                                            min_dist=50., max_dist=80.)
            positions = positions.numpy()
            positions = np.squeeze(positions, axis=0)

            for i,p in enumerate(positions):
                rx = Receiver(name=f"rx-{i}",
                            position=p,
                            orientation=[0,0,0],
                            display_radius=2.0)
                scene.add(rx)

            scene.preview();

        .. figure:: ../figures/rm_mesh_user_sample.png
            :align: center

        The above example shows an example for random positions between 50m and
        80m from the transmitter and a minimum path gain of -100 dB.
        Keep in mind that the transmitter can have a different height than the
        radio map which also contributes to this distance.
        For example if the transmitter is located 20m above the surface of the
        radio map and a ``min_dist`` of 20m is selected, also positions
        directly below the transmitter are sampled.

        :param num_pos: Number of returned random positions for each transmitter

        :param metric: Metric to be considered for sampling positions
        :type metric: "path_gain" | "rss" | "sinr"

        :param min_val_db: Minimum value for the selected metric ([dB] for path
            gain and SINR; [dBm] for RSS).
            Positions are only sampled from cells where the selected metric is
            larger than or equal to this value. Ignored if `None`.

        :param max_val_db: Maximum value for the selected metric ([dB] for path
            gain and SINR; [dBm] for RSS).
            Positions are only sampled from cells where the selected metric is
            smaller than or equal to this value.
            Ignored if `None`.

        :param min_dist:  Minimum distance [m] from transmitter for all random
            positions. Ignored if `None`.

        :param max_dist: Maximum distance [m] from transmitter for all random
            positions. Ignored if `None`.

        :param tx_association: If `True`, only positions associated with a
            transmitter are chosen, i.e., positions where the chosen metric is
            the highest among all all transmitters. Else, a user located in a
            sampled position for a specific transmitter may perceive a higher
            metric from another TX.

        :param center_pos: If `True`, all returned positions are sampled from
            the cell center (i.e., the grid of the radio map). Otherwise, the
            positions are randomly drawn from the surface of the cell.

        :return: Random positions :math:`(x,y,z)` [m]
            (shape: :py:class:`[num_tx, num_pos, 3]`) that are in cells
            fulfilling the configured constraints

        :return: Cell indices (shape :py:class:`[num_tx, num_pos]`)
            corresponding to the random positions
        """

        sampled_cells = super().sample_cells(num_pos,
                                            metric,
                                            min_val_db, max_val_db,
                                            min_dist, max_dist,
                                            tx_association,
                                            seed)

        # If set to True,samples the positions from within the primitive
        if center_pos:
            cell_centers = self.cell_centers
            sampled_pos = dr.gather(mi.Point3f, cell_centers,
                                    dr.ravel(sampled_cells))
            sampled_pos = dr.reshape(mi.TensorXf, sampled_pos,
                                    [self.num_tx, num_pos, 3])
        else:
            # Reset sampler
            self._sampler.seed(seed, num_pos*self.num_tx)
            #
            v_ind = self.measurement_surface.face_indices(dr.ravel(sampled_cells))
            # Three vertices of the triangle
            v0 = v_ind.x
            v1 = v_ind.y
            v2 = v_ind.z
            v0 = self.measurement_surface.vertex_position(dr.ravel(v0))
            v1 = self.measurement_surface.vertex_position(dr.ravel(v1))
            v2 = self.measurement_surface.vertex_position(dr.ravel(v2))
            # Uniformly sample point within the primitive
            esp = self._sampler.next_2d()
            s = dr.sqrt(esp.x)
            t = esp.y
            # Barycentric coordinates
            p = (1-s)*v0 + s*(1-t)*v1 + s*t*v2
            # Reshape
            sampled_pos = dr.reshape(mi.TensorXf, p.array, [self.num_tx, num_pos, 3])

        return sampled_pos, sampled_cells
