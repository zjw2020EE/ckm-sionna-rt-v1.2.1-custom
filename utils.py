import numpy as np
import drjit as dr

def get_basic_metrics(ck_map):
    """
    Extracts and scales basic radio metrics from the CKMap result.
    
    Args:
        ck_map: The result object from CKMapSolver.
        
    Returns:
        pg_db: Path gain in decibels [dB].
        toa_ns: Time of Arrival in nanoseconds [ns] (invalid cells set to NaN).
        rms_ds_ns: RMS Delay Spread in nanoseconds [ns].
        is_los: Boolean mask for Line-of-Sight visibility.
    """
    # 1. Path Gain to dB: Use a floor of 1e-30 to avoid log(0)
    pg = ck_map.path_gain.numpy()
    pg_db = 10.0 * np.log10(np.maximum(pg, 1e-30))
    
    # 2. ToA Processing: Convert the -1.0 sentinel to NaN and scale to ns
    # Note: ToA was stored as UInt32(ns) then cast to Float(s) in finalize()
    toa = ck_map.toa.numpy()
    toa[toa == -1.0] = np.nan
    toa_ns = toa * 1e9
    
    # 3. RMS Delay Spread scaling to ns
    rms_ds_ns = ck_map.rms_ds.numpy() * 1e9
    
    # 4. LoS visibility (Boolean)
    is_los = ck_map.is_los.numpy()
    
    return pg_db, toa_ns, rms_ds_ns, is_los

def get_angular_metrics(mean_vec_tensor):
    """
    Calculates Azimuth, Elevation, and Angular Spread from mean direction vectors.
    
    Mathematical Logic:
    - Directional vectors are power-weighted averages of all paths in a cell.
    - Magnitude ||v|| < 1 indicates angular dispersion (multipath richness).
    - Angular Spread (Spread) = sqrt(1 - ||v||^2).
    """
    # Convert DrJit tensor to Numpy
    vec = mean_vec_tensor.numpy()
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    
    # Calculate vector magnitude for masking and spread calculation
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    
    # Masking: Identify cells with no significant signal to avoid noise
    mask = magnitude < 1e-6
    
    # Calculate Azimuth and Elevation (Results in Radians)
    # Azimuth: Angle in the XY plane
    azi = np.arctan2(y, x)
    # Elevation: Angle relative to the XY plane
    xy_dist = np.sqrt(x**2 + y**2)
    ele = np.arctan2(z, xy_dist)
    
    # Calculate Angular Spread (DSA or DSD)
    # norm_v represents the coherence of the directions
    norm_v = np.clip(magnitude, 0.0, 1.0)
    spread = np.sqrt(np.maximum(1.0 - norm_v**2, 0.0))
    
    # Apply mask to clean up the data for visualization
    azi[mask] = np.nan
    ele[mask] = np.nan
    spread[mask] = np.nan
    
    return azi, ele, spread