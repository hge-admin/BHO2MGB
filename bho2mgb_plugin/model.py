
import numpy as np
import operator
from qgis.core import*
import processing
from osgeo import gdal,osr,ogr
from qgis.utils import iface
import operator
import os
from qgis.PyQt.QtCore import QVariant
import pandas as pd
from .rasterstats_gdal import *

from PyQt5.QtWidgets import QApplication


"""
Isso aqui só está aqui ainda porque eu não resolvi umas coisas de diretório do código.
Se tiver alguma problema é só colocar o endereço local do plugin no teu pc.
"""
#os.chdir()
# =============================================================================
# Module 1
# =============================================================================

pd2ogr = {  'int64': ogr.OFTInteger,
            'int32': ogr.OFTInteger,
            'float64': ogr.OFTReal,
            'object': ogr.OFTString  }

def shp2pd(vector_file):

    layer = QgsVectorLayer(vector_file)

    field_data = layer.dataProvider()
    field_names = [field.name() for field in field_data.fields()]
    features_layer = [feat for feat in layer.getFeatures()]

    df = pd.DataFrame()

    QApplication.processEvents()

    for name_fature in field_names:
        list_feat = [feature_list[name_fature] for feature_list in features_layer]
        df[name_fature] = list_feat
        
    layer = None
    
    return df

def load_df(fn, fields=None):

    ds = ogr.Open(fn)
    lyr = ds.GetLayer()
    lyrDefn = lyr.GetLayerDefn()

    df = pd.DataFrame()

    QApplication.processEvents()

    for i in range(lyrDefn.GetFieldCount()):
        field_name = lyrDefn.GetFieldDefn(i).GetName()
        field_values = []
        for feat in lyr:
            field_values.append(feat.GetField(i))
            geom = feat.GetGeometryRef()
        field_series = pd.Series(field_values, name=field_name)
        df[field_name] = field_series
    if fields:
        df = df[fields]

    for i, feat in enumerate(lyr):
        geom = feat.GetGeometryRef()
        df.loc[i, 'geometry'] = geom.ExportToWkt()
        df.loc[i, 'centroid_x'] = geom.Centroid().GetX() #.ExportToWkt()
        df.loc[i, 'centroid_y'] = geom.Centroid().GetY() #.ExportToWkt()
    try:
        df = df.sort_values(by='cobacia').reset_index(drop=True)
    except:
        None
    return df

def save_df(df, fn, geomtype=ogr.wkbMultiLineString):

    geometry = df.geometry
    df = df.drop('geometry', axis=1)

    # set up the shapefile driver
    driver = ogr.GetDriverByName("ESRI Shapefile")

    # create the data source
    if os.path.exists(fn):
        driver.DeleteDataSource(fn)

    data_source = driver.CreateDataSource(fn)
    #print('data source')
    #print(data_source)
    # create the spatial reference, WGS84
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    #print('proj')
    #print(srs)
    # create the layer
    layer = data_source.CreateLayer('0', srs)
    
    QApplication.processEvents()

    # Add the fields we're interested in
    for name, c in df.iteritems():
        layer.CreateField(ogr.FieldDefn(name, pd2ogr[c.dtype.name]))

    # Process df and add the attributes and features to the shapefile
    for j, row in df.iterrows():
        feature = ogr.Feature(layer.GetLayerDefn())
        #print(row.iteritems())
        #print(feature)
        for i, value in row.iteritems():
            #print(feature.SetField(str(i), str('test')))
            feature.SetField(i, value)

        # create the WKT for the feature using Python string formatting
        wkt = geometry[j]

        # Create geometry from the Well Known Txt
        geom = ogr.CreateGeometryFromWkt(wkt)

        # Set the feature geometry using the point
        feature.SetGeometry(geom)
        # Create the feature in the layer (shapefile)
        layer.CreateFeature(feature)
        # Dereference the feature
        feature = None

    # Save and close the data source
    data_source = None
    layer = None

def pols2lines(polsfn, linesfn):
    
    ds = ogr.Open(polsfn)
    lyr = ds.GetLayer()
    # geometryCollection
    geomcol =  ogr.Geometry(ogr.wkbGeometryCollection)
    # geomcol =  ogr.Geometry(ogr.wkbLinearRing)
    for feat in lyr:
        geom = feat.GetGeometryRef()
        geom_name = geom.GetGeometryName()
        while geom_name in ('POLYGON', 'MULTIPOLYGON'): # if polygon, convert to outlines
            geom = geom.GetGeometryRef(0)
            geom_name = geom.GetGeometryName()
        geomcol.AddGeometry(geom)
    
    # Creates file
    Driver = ogr.GetDriverByName("ESRI Shapefile")
    outDataSource = Driver.CreateDataSource(linesfn)
    outLayer = outDataSource.CreateLayer('', lyr.GetSpatialRef(), geom_type=ogr.wkbMultiLineString)
    featureDefn = outLayer.GetLayerDefn()
    outFeature = ogr.Feature(featureDefn)
    outFeature.SetGeometry(geomcol)
    outLayer.CreateFeature(outFeature)
    
    outDataSource = None
    outLayer = None

def carrega_bho(file, rasterfn, name_file):

    vectorlayer = QgsVectorLayer(file)

    src = gdal.Open(rasterfn)
    ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)

    feature = QgsFeature()
    feature.setGeometry(QgsGeometry.fromRect((QgsRectangle(ulx,uly,lrx,lry))))

    lyr_geom = QgsVectorLayer('Polygon?crs=epsg:4326','Test',"memory")
    lyr_geom.startEditing()
    lyr_geom.addFeature(feature)
    lyr_geom.commitChanges()

    output_file = os.getcwd()

    if not os.path.exists(output_file):
        os.makedirs(output_file)

    output_shp_file = os.path.join(output_file, str(name_file)+'.shp')

    processing.run('native:clip', {'INPUT':vectorlayer,\
                'OVERLAY':lyr_geom,\
                'OUTPUT':output_shp_file})

    df = load_df(output_shp_file)

    return df, output_shp_file

def coords_in_bho(coords_list, bho_areas):

    point_geom = QgsVectorLayer('Point?crs=epsg:4326','Test',"memory")

    feat = QgsFeature()
    point_geom.startEditing()
    for n in range(len(coords_list)):

        feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(coords_list[n][0],coords_list[n][1])))
        point_geom.addFeature(feat)

    point_geom.commitChanges()

    Prj = QgsProject().instance() # Object for current project
    Prj.addMapLayers([point_geom])

    vector_layer = QgsVectorLayer(bho_areas)

    current_dir = os.getcwd()
    #print(current_dir)
    output_file = os.path.join(current_dir, 'sjoin_point.shp')

    processing.run("native:joinbynearest",\
    {'INPUT':point_geom,\
    'INPUT_2':vector_layer,\
    'FIELDS_TO_COPY':[],\
    'DISCARD_NONMATCHING':True,\
    'NEIGHBORS':1,\
    'OUTPUT':output_file})

    cobacias = shp2pd(output_file)['cobacia']
    QgsVectorFileWriter.deleteShapeFile(output_file)
    
    #iface.addVectorLayer(output, '', 'ogr')

    return cobacias

def upstream_bho(cod, df_bho_trecs):
    ''' a partir de um codigo, sobe a rede de trechos da BHO-ANA
            inclui o codigo na lista tbm.
    '''

    #Prepara fc para processamento
    df_work = df_bho_trecs

    trec_out = df_work[df_work['cobacia']==cod] # Trecho mais de jusante

    wc_out = trec_out['cocursodag'].iloc[0] # Curso d'agua desse trecho

    df_out = df_work[(df_work['cocursodag'].str.startswith(wc_out))
                     & (df_work['cobacia']>=cod)]

    return df_out['cobacia'].to_list()

# Sobe rio
def roi_define(df_bho_areas, df_bho_trecs, cods):

    lista_cobacias = upstream_bho(cods[0], df_bho_trecs)
    lista_cobacias = sorted(lista_cobacias)
    sub = [0] * len(lista_cobacias)
    
    QApplication.processEvents()
    
    sub_n = np.arange(len(cods), 0, -1)
    for i, codigo in enumerate(cods):
        lista_i = upstream_bho(codigo, df_bho_trecs)
        sub = np.where(np.isin(lista_cobacias, lista_i), sub_n[i], sub)

    # Filtra os shapes para a area de interesse
    roi_df_areas = df_bho_areas[df_bho_areas['cobacia'].isin(
                    lista_cobacias)].sort_values(by='cobacia').reset_index(drop=True)
    roi_df_areas['sub'] = sub

    roi_df_trecs = df_bho_trecs[df_bho_trecs['cobacia'].isin(
                    lista_cobacias)].sort_values(by='cobacia').reset_index(drop=True)
    roi_df_trecs['sub'] = sub

    # Extrai os pontos dentro da bacia de interesse
    noorigem = list(roi_df_trecs['noorigem'])
    nodestino = list(roi_df_trecs['nodestino'])
    lista_pontos = list(set(noorigem + nodestino))

    #output save
    current_dir = os.getcwd()
    out_roi_areas = os.path.join(current_dir, 'roi_areas.shp')
    #print(out_roi_areas)
    out_roi_trecs = os.path.join(current_dir, 'roi_trecs.shp')
    #print(out_roi_trecs)
    save_df(roi_df_areas[['sub','cotrecho','cobacia',
                            'nuareacont', 'cocursodag','centroid_x','centroid_y',
                            'geometry']], out_roi_areas, ogr.wkbPolygon)
    save_df(roi_df_trecs[['sub','cotrecho','cobacia','nucomptrec',
                            'nuareacont','nuareamont','nutrjus', 'cocursodag','nustrahler','centroid_x','centroid_y',
                            'geometry']], out_roi_trecs, ogr.wkbMultiLineString)
    
    return out_roi_areas, out_roi_trecs

###### NUMPY VERSION #########
def bho2mini(roi_df_trecs, roi_df_areas, uparea_min, lmin):
    
    # Added code to run step 2 directly
    try:
        roi_df_trecs['sub']
    except:
        roi_df_trecs['sub'] = 1
        roi_df_areas['sub'] = 1
    
    lmin = float(lmin)
    # Attributes to use
    sub = roi_df_trecs['sub'].to_numpy()
    cotrecho = roi_df_trecs['cotrecho'].to_numpy()
    cobacia = roi_df_trecs['cobacia'].to_numpy()
    cocursodag = roi_df_trecs['cocursodag'].to_numpy()
    nucomptrec = roi_df_trecs['nucomptrec'].to_numpy()
    nuareacont = roi_df_trecs['nuareacont'].to_numpy()
    nuareamont = roi_df_trecs['nuareamont'].to_numpy()
    nutrjus = roi_df_trecs['nutrjus'].to_numpy()
    # noorig = roi_df_trecs['noorigem'].to_numpy()
    # nodest = roi_df_trecs['nodestino'].to_numpy()

    idx_amon = nuareamont > float(uparea_min)
    subs = np.unique(sub)
    water_courses = np.unique(cocursodag[idx_amon])
    cobacia_agg = cobacia.copy()
    cobacia_agg[:] = '-1'

    QApplication.processEvents()
    for s in subs:
        for c in water_courses:

            idx_twc = (sub == s) & (cocursodag == c) & idx_amon
            cods = cobacia[idx_twc] # This variable will be changed
            cots = cotrecho[idx_twc] # This variable is static among wc domain
            lens = nucomptrec[idx_twc] # trecs in water course (legths)

            cobacia_agg[np.where(np.isin(cobacia, cods))] = cods

            # Down to Up
            idx_min = 0
            while idx_min < len(lens):

                lens_min = lens[idx_min]
                if lens_min < lmin: # if trec length smaller than threshold (needs to be aggregated)
                    idx_domain = sorted(np.where(lens==lens_min)[0]) # indexes all trecs bellonging to aggregation domain

                    if idx_domain[0] == 0: # if most downstream trec
                        lens_jus = 999 # assign downstream to really large number
                    else:
                        lens_jus = lens[idx_domain[0]-1] # assign downstream as real downstream

                    if idx_min == len(lens)-1: # if last trec in the water course domain
                        # try alongating upstream trec using outside domain
                        idx_mon = (cocursodag == c) & (nutrjus == cots[idx_min])
                        try:
                            lens_out = float(nucomptrec[idx_mon])
                        except:
                            lens_out = 999

                        if lens_out < lens_jus: # Alongates only if it will be part of wc domain (minimum neighbour)
                            # updates lists related to water course
                            lens = np.append(lens, lens_out)
                            cods = np.append(cods, cobacia[idx_mon])
                            cots = np.append(cots, cotrecho[idx_mon])
                            cobacia_agg[idx_mon] = cobacia[idx_mon]

                    try:
                        lens_mon = lens[idx_domain[-1]+1] # assign (if exists) next trec outside twc domain
                    except:
                        lens_mon = 999 # assign upstream to really large number

                    if (lens_mon != 999) | (lens_jus != 999): # If has trec down or up
                        lens_agg = np.minimum(lens_mon, lens_jus) # trec which will be aggregated (min neighbour)
                        idx_agg = np.where(lens==lens_agg)[0][0] # indexing trec

                        agg_cods = sorted(cods[[idx_min, idx_agg]]) # takes first cobacia as aggregation code (most downstream)
                        cobacia_agg[np.where(np.isin(cobacia_agg, agg_cods))] = agg_cods[0] # updates indexes
                        cods[np.isin(cods, agg_cods)] = agg_cods[0]
                        lens[np.isin(cods, agg_cods)] = lens_min + lens_agg

                idx_min += 1

    # Assign areas to main trecs
    midx = (-np.ones(cobacia.shape)) # trec index (start with -1 to later assign)
    mcods = np.delete(np.unique(cobacia_agg), np.unique(cobacia_agg)=='-1') # cods of minis
    mcot = midx.copy().astype('int')
    mjus = midx.copy().astype('int')
    mlen = midx.copy()
    mareacont = midx.copy()
    mareamont = midx.copy()
    # morig = midx.copy()
    # mdest = midx.copy()
    for i, mcod in enumerate(mcods[::-1]): # do from upstream to downstream
        idx_mcod = np.where(cobacia==mcod)[0][0] # most upstream cobacia
        # Get trecs belong to mini by mcod
        mcods_in = (np.char.startswith(cocursodag.astype(str), cocursodag[idx_mcod]) # upstream water course filter
                     & (cobacia >= mcod) # upstream cobacia filter
                     & (midx == -1) ) # trecs not yet assigned, which are the more downstream ones (in trec)
        midx[mcods_in] = i # Assign mini code

        # Get trec and trecjus
        idx_main = np.where(cobacia_agg==mcod)[0] # main trec
        mcot[mcods_in] = cotrecho[idx_main[-1]]
        mjus[np.where(np.isin(nutrjus, cotrecho[idx_main]))] = cotrecho[idx_main[-1]]

        # Get length and areas
        mlen[mcods_in] = nucomptrec[idx_main].sum()
        mareacont[mcods_in] = nuareacont[mcods_in].sum()
        mareamont[mcods_in] = nuareamont[idx_main[0]]

        mcods_up = mcods_in # upstream trec for next iteration

        # Get draining points
        # idx_main = np.where(cobacia_agg==mcod)[0] # main trec
        # morig[mcods_in] = noorig[idx_main[-1]]
        # mdest[mcods_in] = nodest[idx_main[0]]

    #QApplication.processEvents()

    # mtrecs df
    mtrecs = roi_df_trecs[['sub', 'cocursodag', 'cobacia', 'geometry']]
    # mtrecs['morig'] = morig
    # mtrecs['mdest'] = mdest
    mtrecs['agg'] = cobacia_agg
    mtrecs['cotrecho'] = mcot
    mtrecs['nutrjus'] = mjus
    mtrecs['nucomptrec'] = mlen
    mtrecs['nuareacont'] = mareacont
    mtrecs['nuareamont'] = mareamont
    mtrecs = mtrecs[mtrecs['agg'] != '-1']

    # mareas df
    mareas = roi_df_areas[['sub', 'cocursodag', 'cobacia', 'geometry']]
    mareas['midx'] = midx
    mareas['cotrecho'] = mcot
    mareas['nutrjus'] = mjus
    mareas['nucomptrec'] = mlen
    mareas['nuareacont'] = mareacont
    mareas['nuareamont'] = mareamont
    
    # Save minis BHO with mini index
    bho_midx = mareas[['sub', 'midx', 'cobacia', 'geometry']]
    save_df(bho_midx, 'bho2mini.shp', ogr.wkbPolygon)

    current_dir = os.getcwd()
    
    output_file_mtrecs = os.path.join(current_dir, 'mtrecs.shp')
    dissolver(mtrecs, 'agg', output_file_mtrecs,  ogr.wkbLineString)
    
    output_file_mareas = os.path.join(current_dir, 'mareas.shp')
    dissolver(mareas, 'midx', output_file_mareas,  ogr.wkbPolygon)

    return output_file_mtrecs, output_file_mareas

def dissolver(df, by, outfn, geomtype):

    geometry = df[[by, 'geometry']]

    # Dissolve geometry
    current_dir = os.getcwd()
    input_geom = os.path.join(current_dir, 'geom2dissolve.shp')
    save_df(geometry, input_geom, geomtype)
    
    df = df.drop('geometry', axis=1)
    # geomagg = geometry.groupby(by).agg('; '.join)
    dfagg = df.groupby(by).first().reset_index()
    
    geom_layer = QgsVectorLayer(input_geom,'','ogr')
    out_dissolve_geom =  os.path.join(current_dir, 'geomdissolved.shp')
    layer_dissolve = processing.run("native:dissolve",
             {'INPUT': geom_layer,
             'FIELD': [by],
             'OUTPUT': out_dissolve_geom})
    geom_layer = None
    
    dissolve_geom = load_df(out_dissolve_geom).sort_values(by=by).reset_index()
    dfagg['geometry'] = dissolve_geom['geometry']
    dfagg = dfagg.sort_values(by='cobacia').reset_index(drop=True)
    out_dissolve_df =  os.path.join(current_dir, outfn)
    save_df(dfagg, out_dissolve_df, geomtype)
    
    QgsVectorFileWriter.deleteShapeFile(input_geom)
    QgsVectorFileWriter.deleteShapeFile(out_dissolve_geom)
    
    return None

# =============================================================================
# Get river slope in trecss
# =============================================================================
def get_slopes_main(mtrecsfn, demfn):

    # dem = gdal.Open(demfn)
    # #mxz = dem.ReadAsArray()
    # nodatavalue = dem.GetRasterBand(1).GetNoDataValue()

    # Compute slopes on main stretches
    elvs = zonal_stats(
        mtrecsfn,
        demfn, stats=['percentile_10', 'percentile_85'])
    elv10 = np.array([d['percentile_10'] for d in elvs])
    elv85 = np.array([d['percentile_85'] for d in elvs])
    dh = elv85 - elv10

    # Opened in geopandas!!!!!!
    #mtrecs = gpd.read_file(mtrecsfn)
    mtrecs = shp2pd(mtrecsfn)
    ltrecs = mtrecs['nucomptrec'].to_numpy() * 0.75
    slopes_main = dh / ltrecs

    return slopes_main

def get_slopes_afl(mareasfn, handfn, ltndfn):

    mem_driver = ogr.GetDriverByName("Memory")
    mem_driver_gdal = gdal.GetDriverByName("MEM")
    shp_name = "temp"

    hand_ds = gdal.Open(handfn)
    ltnd_ds = gdal.Open(ltndfn)
    p_ds = ogr.Open(mareasfn)

    lyr = p_ds.GetLayer()
    geot = hand_ds.GetGeoTransform()
    #nodata = ltnd_ds.GetRasterBand(1).GetNoDataValue()

    len_afl = []
    slopes_afl = []

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

            hand_array = hand_ds.GetRasterBand(1).ReadAsArray(
            offsets[2],
            offsets[0],
            offsets[3] - offsets[2],
            offsets[1] - offsets[0])

            ltnd_array = ltnd_ds.GetRasterBand(1).ReadAsArray(
            offsets[2],
            offsets[0],
            offsets[3] - offsets[2],
            offsets[1] - offsets[0])

            mask_hand = np.ma.MaskedArray(
                hand_array,
                mask=np.logical_or(np.isnan(hand_array), np.logical_not(tr_array)))
            mask_ltnd = np.ma.MaskedArray(
                ltnd_array,
                mask=np.logical_or(np.isnan(ltnd_array), np.logical_not(tr_array)))
            # maskarray = maskarray[~np.isnan(maskarray)]

            lmax = mask_ltnd.max()
            dh = np.mean(mask_hand[np.where(mask_ltnd==lmax)])

            len_afl.append(lmax)
            slopes_afl.append(dh/lmax)

            tp_ds = None
            tp_lyr = None
            tr_ds = None

            p_feat = lyr.GetNextFeature()

    return slopes_afl, len_afl

# =============================================================================
# Get HRUs
# =============================================================================
def get_hrus(mareasfn, hrufn):

    hru_rate = zonal_stats(
                mareasfn,
                hrufn,categorical=True)
    n = pd.DataFrame(hru_rate).columns.max()
    hru_df = pd.DataFrame(hru_rate, columns=np.arange(1,n+1,1))
    hru_df.columns = hru_df.columns.map('{:0>2d}'.format)
    hru_df = hru_df.add_prefix('BLC_')

    hru_df = hru_df.divide(hru_df.sum(axis=1), axis=0) * 100
    hru_df[np.isnan(hru_df)] = 0

    return hru_df

# =============================================================================
# Informations to write MINI
# =============================================================================
def write_mini(mtrecs, mareas, hru_df, georel, smin, smax, nman):
    
    ## Compute order
    mtrecs['order'] = 0
    for i, trec in mtrecs[::-1].iterrows():
        order = np.nansum([mtrecs[mtrecs['nutrjus'] == trec['cotrecho']]['order'].max(), 1], dtype=int)
        mtrecs.loc[i, 'order'] = order
    
    ## Stretches
    cobacia = mtrecs['cobacia']
    cotrecho = mtrecs['cotrecho']
    centroid_x = mareas['centroid_x']
    centroid_y = mareas['centroid_y']
    geometry = mareas['geometry']
    
    sub = mtrecs['sub']
    hdr = pd.Series([0]*len(sub))
    area_unit = mtrecs['nuareacont']
    area_upstream = mtrecs['nuareamont']
    river_l = mtrecs['nucomptrec']
    river_s = np.minimum(np.maximum(mtrecs['nudecltrec'], smin), smax)
    afl_l = mtrecs['nucompafl']
    afl_s = np.minimum(np.maximum(mtrecs['nudeclafl'], smin), smax)
    trechojus = mtrecs['nutrjus']
    ordem = mtrecs['order']
    nmans = pd.Series([nman]*len(sub))

    a = georel['a']; b = georel['b']; c = georel['c']; d = georel['d'];
    river_b = a * area_upstream ** b
    river_h = c * area_upstream ** d

    infos_df = pd.DataFrame({
            'DN': cobacia,
            'ID_Mini': cotrecho,
            'X_Cen':centroid_x,
            'Y_Cen': centroid_y,
            'Sub_Basin': sub,
            'Area': area_unit,
            'Upst_Area': area_upstream,
            'Reach_Len': river_l,
            'Reach_Slp': river_s,
            'Tribut_Len': afl_l,
            'Tribut_Slp': afl_s,
            'Mini_Jus': trechojus,
            'Order': ordem,
            'Hdr': hdr,
            'Width': river_b,
            'Depth': river_h,
            'NMAN': nmans,
            'geometry': geometry
            })

    mini_df = pd.concat([infos_df, hru_df], axis=1)

    mini_df.sort_values(by=['Order', 'DN'], ascending=[True, False], inplace=True)

    # Change minijus to ordered 1:N
    mini_df.index = np.arange(1,len(cotrecho)+1)
    minijus = []
    for _, row in mini_df.iterrows():
        jus = mini_df.loc[mini_df['ID_Mini']==row['Mini_Jus']].index.values
        minijus.extend(jus)
    minijus.extend([-1])

    mini_df['DN'] = mini_df['ID_Mini']
    mini_df['ID_Mini'] = mini_df.index.values
    mini_df['Mini_Jus'] = minijus

    # Write mini / por algum motivo o .to_string adiciona 2 espacos entre os valores, por isso subtrair por 2
    # f5 = '{:>3}'.format
    # f8 = '{:>6}'.format
    # f10 = '{:>8}'.format
    # f15 = '{:>13}'.format
    mini2txt = mini_df.rename(columns={
            'DN': 'CatID',
            'ID_Mini': 'Mini',
            'X_Cen': 'Xcen',
            'Y_Cen': 'Ycen',
            'Sub_Basin': 'Sub',
            'Area': 'Area_(km2)',
            'Upst_Area': 'AreaM_(km2)',
            'Reach_Len': 'Ltr_(km)',
            'Reach_Slp': 'Str_(m/km)',
            'Tribut_Len': 'Lrl_(km)',
            'Tribut_Slp': 'Srl_(m/km)',
            'Mini_Jus': 'MiniJus',
            'Order': 'Ordem'})
    mini_txt = mini2txt.drop('geometry', axis=1).to_string(
        index=False, formatters = {
            'CatID': '{:>8.0f}'.format,
            'Mini': '{:>7.0f}'.format,
            'Xcen': '{:>13.5f}'.format,
            'Ycen': '{:>13.5f}'.format,
            'Sub': '{:>7}'.format,
            'Area_(km2)': '{:>13.5f}'.format,
            'AreaM_(km2)': '{:>13.5f}'.format,
            'Ltr_(km)': '{:>13.5f}'.format,
            'Str_(m/km)': '{:>13.5f}'.format,
            'Lrl_(km)': '{:>13.5f}'.format,
            'Srl_(m/km)': '{:>13.5f}'.format,
            'MiniJus': '{:>6.0f}'.format,
            'Ordem': '{:>6.0f}'.format,
            'Hdr': '{:>3.0f}'.format,
            'Width': '{:>8.2f}'.format,
            'Depth': '{:>8.2f}'.format,
            'NMAN': '{:>8.4f}'.format,
            }, 
        float_format='{:>8.5f}'.format)
    
    return mini_df, mini_txt

# =============================================================================
# # Write cota-area
# =============================================================================
def flood_area(mareasfn, handfn, mtrecsfn):

    mem_driver = ogr.GetDriverByName("Memory")
    mem_driver_gdal = gdal.GetDriverByName("MEM")
    shp_name = "temp"

    hand_ds = gdal.Open(handfn)
    p_ds = ogr.Open(mareasfn)

    lyr = p_ds.GetLayer()
    geot = hand_ds.GetGeoTransform()
    #nodata = hand_ds.GetRasterBand(1).GetNoDataValue()

    flood_count = []
    cell_count = []

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

            hand_array = hand_ds.GetRasterBand(1).ReadAsArray(
            offsets[2],
            offsets[0],
            offsets[3] - offsets[2],
            offsets[1] - offsets[0])

            hand_class = np.copy(hand_array)
            hand_class[hand_array>100] = np.nan
            for i in np.arange(100, 0, -1):
                hand_class[hand_array<=i] = i

            mask_area = np.ma.MaskedArray(
                hand_array,
                mask=np.logical_or(np.isnan(hand_array), np.logical_not(tr_array)))
            mask_class = np.ma.MaskedArray(
                hand_class,
                mask=np.logical_or(np.isnan(hand_class), np.logical_not(tr_array)))

            flood_count.append(
                setFeatureStats(mask_class, 'count', categorical=True)
                )
            cell_count.append(
                setFeatureStats(mask_area, 'count')
                )

            tp_ds = None
            tp_lyr = None
            tr_ds = None

            p_feat = lyr.GetNextFeature()

    flood_count = pd.DataFrame(flood_count).drop('count', axis=1).sort_index(axis=1)
    flood_count = flood_count.cumsum(axis=1).fillna(method='ffill', axis=1)

    cell_count = pd.Series(pd.DataFrame(cell_count)['count'])

    # mareas = gpd.read_file(mareasfn) # !!!!!!! GEOPANDAS

    mareas_df = load_df(mtrecsfn)

    mini_area = mareas_df['nuareacont']

    flood_area = flood_count.divide(cell_count/mini_area, axis=0)

    return flood_area

def write_cota_area(mareasfn, mtrecsfn, demfn, handfn):

    fa = flood_area(mareasfn, handfn, mtrecsfn)

    z0 = zonal_stats(
            mtrecsfn,
            demfn, stats='median')
    z0 = pd.Series(pd.DataFrame(z0)['median'])


    mtrecs = load_df(mtrecsfn)
    ## Compute order (Redundance! Can be optimized, but doesn't affect performance much)
    mtrecs['order'] = 0
    for i, trec in mtrecs[::-1].iterrows():
        order = np.nansum([mtrecs[mtrecs['nutrjus'] == trec['cotrecho']]['order'].max(), 1], dtype=int)
        mtrecs.loc[i, 'order'] = order

    # Organize dataframe to correspond to MINI order
    mini_flood = fa.copy()
    mini_flood['z0'] = z0
    mini_flood[['ordem', 'cobacia']] = mtrecs[['order', 'cobacia']]
    mini_flood.sort_values(by=['ordem', 'cobacia'], ascending=[True, False], inplace=True)
    mini_flood.drop(['ordem', 'cobacia'], axis=1, inplace=True)
    mini_flood.index = np.arange(1, len(mini_flood)+1)

    # Write Cota-Area
    ca_df = pd.DataFrame(columns = ['mini', 'z0', 'zfp', 'afp'])
    for _, mini in mini_flood.iterrows():
        afp = mini.drop(['z0']).astype(float)
        zi = pd.Series(mini['z0']).repeat(len(afp)).astype(int)
        zf = (afp.index + zi).astype(int)
        mini_i = pd.Series(mini.name).repeat(len(afp)).astype(int)

        mini_i.reset_index(drop=True, inplace=True)
        zi.reset_index(drop=True, inplace=True)
        zf.reset_index(drop=True, inplace=True)
        afp.reset_index(drop=True, inplace=True)

        ca_mini = pd.DataFrame({'mini': mini_i,
                                'z0': zi,
                                'zfp': zf,
                                'afp': afp})

        ca_df = ca_df.append(ca_mini, ignore_index=True)

    ca_txt = ca_df.to_string(index=False, header=False,
                             formatters = {'mini': '{:>9}'.format,
                                           'z0': '{:>8}'.format,
                                           'zfp': '{:>8}'.format,
                                           'afp': '{:>9.2f}'.format})

    ca_txt = ' Minibacia           Z0(m)           Zfp(m)          Areafp(km2)\n' + ca_txt

    return ca_txt