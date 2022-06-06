# -*- coding: utf-8 -*-
"""
zonal_stats function

Adapted from https://towardsdatascience.com/zonal-statistics-algorithm-with-python-in-4-steps-382a3b66648a
and from the rasterstats python package (to run with gdal)

Date: 14 June 2021
"""
from osgeo import gdal, ogr
import os
import numpy as np
import csv


def get_percentile(stat):
    if not stat.startswith('percentile_'):
        raise ValueError("must start with 'percentile_'")
    qstr = stat.replace("percentile_", '')
    q = float(qstr)
    if q > 100.0:
        raise ValueError('percentiles must be <= 100')
    if q < 0.0:
        raise ValueError('percentiles must be >= 0')
    return q


def check_stats(stats, categorical):
    DEFAULT_STATS = ['count', 'min', 'max', 'mean']
    VALID_STATS = DEFAULT_STATS + \
    ['sum', 'std', 'median', 'majority', 'minority', 'unique', 'range', 'nodata', 'nan']

    if not stats:
        if not categorical:
            stats = DEFAULT_STATS
        else:
            stats = []
    else:
        if isinstance(stats, str):
            if stats in ['*', 'ALL']:
                stats = VALID_STATS
            else:
                stats = stats.split()
    for x in stats:
        if x.startswith("percentile_"):
            get_percentile(x)
        elif x not in VALID_STATS:
            raise ValueError(
                "Stat `%s` not valid; "
                "must be one of \n %r" % (x, VALID_STATS))

    run_count = False
    if categorical or 'majority' in stats or 'minority' in stats or 'unique' in stats:
        # run the counter once, only if needed
        run_count = True

    return stats, run_count


def remap_categories(category_map, stats):
    def lookup(m, k):
        """ Dict lookup but returns original key if not found
        """
        try:
            return m[k]
        except KeyError:
            return k

    return {lookup(category_map, k): v
            for k, v in stats.items()}

def key_assoc_val(d, func, exclude=None):
    """return the key associated with the value returned by func
    """
    vs = list(d.values())
    ks = list(d.keys())
    key = ks[vs.index(func(vs))]
    return key
 
# def setFeatureStats(fid, min, max, mean, median, sd, sum, count, names=["min", "max", "mean", "median", "sd", "sum", "count", "id"]):
#     featstats = {
#     names[0]: min,
#     names[1]: max,
#     names[2]: mean,
#     names[3]: median,
#     names[4]: sd,
#     names[5]: sum,
#     names[6]: count,
#     names[7]: fid,
#     }
#     return featstats

def setFeatureStats(masked, stats,
                    categorical=False, category_map=None):
    stats, run_count = check_stats(stats, categorical)
    
    if masked.compressed().size == 0:
        # nothing here, fill with None and move on
        feature_stats = dict([(stat, None) for stat in stats])
        if 'count' in stats:  # special case, zero makes sense here
            feature_stats['count'] = 0
    else:
        if run_count:
            keys, counts = np.unique(masked.compressed(), return_counts=True)
            try:
                pixel_count = dict(zip([k.item() for k in keys],
                                       [c.item() for c in counts]))
            except AttributeError:
                pixel_count = dict(zip([np.asscalar(k) for k in keys],
                                       [np.asscalar(c) for c in counts]))

        if categorical:
            feature_stats = dict(pixel_count)
            if category_map:
                feature_stats = remap_categories(category_map, feature_stats)
        else:
            feature_stats = {}
    
        
        if 'min' in stats:
            feature_stats['min'] = float(masked.min())
        if 'max' in stats:
            feature_stats['max'] = float(masked.max())
        if 'mean' in stats:
            feature_stats['mean'] = float(masked.mean())
        if 'count' in stats:
            feature_stats['count'] = int(masked.count())
        # optional
        if 'sum' in stats:
            feature_stats['sum'] = float(masked.sum())
        if 'std' in stats:
            feature_stats['std'] = float(masked.std())
        if 'median' in stats:
            feature_stats['median'] = float(np.median(masked.compressed()))
        if 'majority' in stats:
            feature_stats['majority'] = float(key_assoc_val(pixel_count, max))
        if 'minority' in stats:
            feature_stats['minority'] = float(key_assoc_val(pixel_count, min))
        if 'unique' in stats:
            feature_stats['unique'] = len(list(pixel_count.keys()))
        if 'range' in stats:
            try:
                rmin = feature_stats['min']
            except KeyError:
                rmin = float(masked.min())
            try:
                rmax = feature_stats['max']
            except KeyError:
                rmax = float(masked.max())
            feature_stats['range'] = rmax - rmin
        
        masked = masked.compressed()
        for pctile in [s for s in stats if s.startswith('percentile_')]:
            q = get_percentile(pctile)
            #pctarr = masked.compressed()
            feature_stats[pctile] = np.percentile(masked, q)
    
    return feature_stats

def boundingBoxToOffsets(bbox, geot):
    col1 = int((bbox[0] - geot[0]) / geot[1])
    col2 = int((bbox[1] - geot[0]) / geot[1]) + 1
    row1 = int((bbox[3] - geot[3]) / geot[5])
    row2 = int((bbox[2] - geot[3]) / geot[5]) + 1
    return [row1, row2, col1, col2]
 
 
def geotFromOffsets(row_offset, col_offset, geot):
    new_geot = [
    geot[0] + (col_offset * geot[1]),
    geot[1],
    0.0,
    geot[3] + (row_offset * geot[5]),
    0.0,
    geot[5]
    ]
    return new_geot

def zonal_stats(fn_zones, fn_raster,
                    stats=None, categorical=False, category_map=None):
    
    mem_driver = ogr.GetDriverByName("Memory")
    mem_driver_gdal = gdal.GetDriverByName("MEM")
    shp_name = "temp"
     
    r_ds = gdal.Open(fn_raster)
    p_ds = ogr.Open(fn_zones)
     
    lyr = p_ds.GetLayer()
    geot = r_ds.GetGeoTransform()
    nodata = r_ds.GetRasterBand(1).GetNoDataValue()
    
    zstats = []
     
    p_feat = lyr.GetNextFeature()
     
    while p_feat:
        if p_feat.GetGeometryRef() is not None:
            if os.path.exists(shp_name):
                mem_driver.DeleteDataSource(shp_name)
            tp_ds = mem_driver.CreateDataSource(shp_name)
            tp_lyr = tp_ds.CreateLayer('polygons', lyr.GetSpatialRef(), ogr.wkbPolygon)
            tp_lyr.CreateFeature(p_feat.Clone())
            offsets = boundingBoxToOffsets(p_feat.GetGeometryRef().GetEnvelope(), geot)
            new_geot = geotFromOffsets(offsets[0], offsets[2], geot)
             
            tr_ds = mem_driver_gdal.Create(
            "", 
            offsets[3] - offsets[2], 
            offsets[1] - offsets[0], 
            1, 
            gdal.GDT_Byte)
             
            tr_ds.SetGeoTransform(new_geot)
            gdal.RasterizeLayer(tr_ds, [1], tp_lyr, burn_values=[1])
            tr_array = tr_ds.ReadAsArray()
             
            r_array = r_ds.GetRasterBand(1).ReadAsArray(
            offsets[2],
            offsets[0],
            offsets[3] - offsets[2],
            offsets[1] - offsets[0])
            
            #------------------------------------------------------------------
            # Change in original code
            #    called functions copied from the rasterstats module
            try:
                nodata = r_array.dtype.type(nodata)
            except:
                nodata = -9999
            if np.isnan(nodata):
                maskarray = np.ma.MaskedArray(
                     r_array,
                     mask=np.logical_or(np.isnan(r_array), np.logical_not(tr_array)))
            else:
                maskarray = np.ma.MaskedArray(
                    r_array,
                    mask=np.logical_or(r_array==nodata, np.logical_not(tr_array)))                
            
            zstats.append(
                setFeatureStats(maskarray, stats, categorical, category_map)
                )
            
            tp_ds = None
            tp_lyr = None
            tr_ds = None
            
            p_feat = lyr.GetNextFeature()
    
    return zstats




