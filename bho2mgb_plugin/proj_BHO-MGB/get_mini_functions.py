# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:12:36 2020

@author: Rafael
"""

import ee
import numpy as np
import pandas as pd
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import Point
import shapely.ops as sops
from rasterstats import zonal_stats
import rasterio
#import geedownload
#import eeconvert
#import time

ee.Initialize()

#caminho para pastas da BHO
def bho_getfiles(version, folder='', ext='.gpkg'):
    if version == '5k':
        bho_files={
                'area':folder+'geoft_bho_2017_5k_area_drenagem'+ext,
                'trecho':folder+'geoft_bho_2017_5k_trecho_drenagem'+ext,
                'ponto':folder+'geoft_bho_2017_5k_ponto_drenagem'+ext,
                }
    elif version == '50k':
        bho_files={
                'area':folder+'geoft_bho_2017_50k_area_drenagem'+ext,
                'trecho':folder+'geoft_bho_2017_50k_trecho_drenagem'+ext,
                'ponto':folder+'geoft_bho_2017_50k_ponto_drenagem'+ext,
                }
    elif version == '250':
        bho_files={
                'area':folder+'geoft_bho_area_drenagem'+ext,
                'trecho':folder+'geoft_bho_trecho_drenagem'+ext,
                'ponto':folder+'geoft_bho_ponto_drenagem'+ext,
                }
    else:
        print('Provide a valid version (5k or 50k)')
    return bho_files

def carrega_bho(file, rasterfn):
    # Carrega o DEM para filtrar area de interesse (+ rapido p carregar)
    src = gdal.Open(rasterfn)
    ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)

    bbox = (ulx, lry, lrx, uly)

    """ carrega arquivo bho e garante dtypes inteiros em string"""
    gdf = gpd.read_file(file, bbox=bbox)
    # crs = gdf.crs

    # # converte o que foi lido como int64 para object (queremos strings!)
    # d = {k:np.dtype('O') for (k,v) in gdf.dtypes.items() if v=='int64'}

    # # correcao do crs (por algum motivo .astype detona)
    # gdf = gdf.astype(d)
    # gdf.crs = crs

    return gdf

def coords_in_bho(coords_list, df_bho_area):
    ids = np.arange(len(coords_list))
    geometry = [Point(xy) for xy in coords_list]
    points = gpd.GeoDataFrame(ids, crs = df_bho_area.crs, geometry = geometry)

    #aplica o join no shape de pontos (left index)
    pontos_em_pols = gpd.sjoin(points, df_bho_area, how='left',op='within')
    cobacias = pontos_em_pols['cobacia']

    return cobacias

def upstream_bho(codigo, df_bho_trecho):
    ''' a partir de um codigo, sobe a rede de trechos da BHO-ANA
            inclui o codigo na lista tbm.
    '''

    #Prepara fc para processamento
    df_work = df_bho_trecho

    # A subida é realizada por cobacia.
    # Inicia a subida do rio
    cobacia = [codigo]
    next_cobacia = cobacia
    lista_cobacias = []  # lista de subida global
    sobe = 1
    while sobe == 1:

        #subida global
        cobacia = next_cobacia
        lista_cobacias.extend(cobacia)

        #bacia atual
        bacia = df_work[df_work['cobacia'].isin(cobacia)]

        #encontra trecho e bacias logo a montante
        trecho = bacia['cotrecho']
        montante = df_work[df_work['nutrjus'].isin(trecho)]
        next_cobacia = montante['cobacia']

        if len(next_cobacia) == 0: # check if list is empty
            sobe = 0

    return lista_cobacias

# =============================================================================
# # Collect data
# =============================================================================

# Sobe rio
def roi_define(df_bho_area, df_bho_trecho, df_bho_ponto, cods, level=1, method='coord'):

    if method=='coord':
        lista_cobacias = upstream_bho(cods[0], df_bho_trecho)
        lista_cobacias = sorted(lista_cobacias)
        sub = [0] * len(lista_cobacias)
        for i, codigo in enumerate(cods):
            lista_i = upstream_bho(codigo, df_bho_trecho)
            sub = np.where(np.isin(lista_cobacias, lista_i), i+1, sub)

        # Filtra os shapes para a area de interesse
        roi_df_area = df_bho_area[df_bho_area['cobacia'].isin(
                        lista_cobacias)].sort_values(by='cobacia').reset_index(drop=True)
        roi_df_area['sub'] = sub

        roi_df_trecho = df_bho_trecho[df_bho_trecho['cobacia'].isin(
                        lista_cobacias)].sort_values(by='cobacia').reset_index(drop=True)
        roi_df_trecho['sub'] = sub

    elif method=='nunivotto':
        roi_df_area = df_bho_area[df_bho_area['nunivotto' + str(level)].isin(
                        cods)].sort_values(by='cobacia').reset_index(drop=True)
        lista_cobacias = list(roi_df_area['cobacia'])

        roi_df_trecho = df_bho_trecho[df_bho_trecho['cobacia'].isin(
                        lista_cobacias)].sort_values(by='cobacia').reset_index(drop=True)

    # Extrai os pontos dentro da bacia de interesse
    noorigem = list(roi_df_trecho['noorigem'])
    nodestino = list(roi_df_trecho['nodestino'])
    lista_pontos = list(set(noorigem + nodestino))

    roi_df_ponto = (df_bho_ponto[df_bho_ponto['idponto'].isin(
                        lista_pontos)].sort_values(by='idponto').reset_index(drop=True))

    return roi_df_area, roi_df_trecho, roi_df_ponto, lista_cobacias, lista_pontos

# =============================================================================
# Filter BHOS to MINI
# =============================================================================

def bho2mini(roi_df_trecho, roi_df_area, uparea_min = 30, lmin = 6):

    mtrecs = roi_df_trecho[roi_df_trecho['nuareamont']>uparea_min]
    mtrecs = mtrecs[['sub',
                         'cotrecho',
                         'cobacia',
                         'cocursodag',
                         'noorigem',
                         'nodestino',
                         'nucomptrec',
                         'nuareamont',
                         'nutrjus',
                         'nustrahler',
                         'geometry']]
    roi_df_trecho['ismain'] = 0

    water_courses = mtrecs.groupby('cocursodag').size()

    for c in water_courses.index:
        twc = mtrecs[mtrecs['cocursodag']==c].sort_values(by='cobacia')

        while twc['nucomptrec'].min()<lmin:

            tmin = twc[twc['nucomptrec']==twc['nucomptrec'].min()].iloc[0]
            if any(twc['nutrjus']==tmin['cotrecho']): #se existir trechos a montante
                tmin_mon = twc[twc['nutrjus']==tmin['cotrecho']].iloc[0]
            else:
                # Aqui o trecho mais de montante curto é alongado se der
                if any(roi_df_trecho['nutrjus'] == tmin['cotrecho']):
                    tmin_mon = roi_df_trecho[
                        (roi_df_trecho['nutrjus'] == tmin['cotrecho']) &
                        (roi_df_trecho['cocursodag'] == tmin['cocursodag'])].iloc[0]
                else:
                    tmin_mon = tmin.copy()
                    tmin_mon['nucomptrec'] = 9999
            if any(twc['cotrecho']==tmin['nutrjus']): #se existir trechos a jusante
                tmin_jus = twc[twc['cotrecho']==tmin['nutrjus']].iloc[0]
            else:
                tmin_jus = tmin.copy()
                tmin_jus['nucomptrec'] = 9999

            if tmin_jus.name == tmin_mon.name:
                twc.loc[tmin.name, 'nucomptrec'] = 9999
                continue

            l2r = np.minimum(tmin_mon['nucomptrec'], tmin_jus['nucomptrec'])

            if tmin_mon['nucomptrec'] == l2r:
                tmon = tmin_mon
                tjus = tmin
            elif tmin_jus['nucomptrec'] == l2r:
                tmon = tmin
                tjus = tmin_jus
            else:
                print('Error in finding minimum length!')

            t2r = tjus.copy()
            # t2r['cobacia'] = tmon['cobacia']
            t2r['cotrecho'] = tmon['cotrecho']
            t2r['noorigem'] = tmon['noorigem']
            t2r['nucomptrec'] = tjus['nucomptrec'] + tmon['nucomptrec']
            t2r['geometry'] = sops.unary_union([tjus.geometry, tmon.geometry])

            twc.loc[t2r.name] = t2r
            mtrecs.loc[t2r.name] = t2r

            roi_df_trecho.loc[ # Atributte code to original file to track which trec was aggregated
                roi_df_trecho.index.isin([tmon.name, tjus.name]), 'ismain'] = 1

            try:
                twc.drop(tmon.name, inplace=True)
                mtrecs.drop(tmon.name, inplace=True)
            except:
                continue
            #     cobacias_in.loc[t2r.name].extend([tmon['cobacia']])

    mtrecs = mtrecs.sort_values(by='cobacia').reset_index(drop=True)
    mtrecs = mtrecs.set_crs(roi_df_area.crs)
    mtrecs['nucomptrec'] = mtrecs.geometry.to_crs("ESRI:102033").length / 1000

    for i, tr in mtrecs.iterrows():
    # Essa parte nao ta nem um pouco otimizada...
    # Nao aumenta o tempo consideravelmente, mas da pra melhorar
        cobacias_upstr = upstream_bho(tr['cobacia'], roi_df_trecho)
        trecs_upstr = roi_df_trecho['cobacia'].isin(cobacias_upstr)
        roi_df_trecho.loc[
            trecs_upstr, 'midx'] = i

        # cobacias_upstr.remove(tr['cobacia'])
        # trecs_upstr = roi_df_trecho['cobacia'].isin(cobacias_upstr)
        # mtrecs.loc[
        #     mtrecs['cotrecho'].isin(roi_df_trecho.loc[trecs_upstr, 'cotrecho']),
        #     'nutrjus'] = tr['cotrecho']

    roi_df_area['midx'] = roi_df_trecho['midx']
    mpols = roi_df_area[['sub', 'midx', 'cobacia', 'cocursodag', 'geometry']]
    mpols = mpols.dissolve(by='midx', aggfunc='first')
    mpols['nuareacont'] = mpols.geometry.to_crs("ESRI:102033").area / 1000000

    for i, tr in mtrecs.iterrows():
        tmini = roi_df_trecho[roi_df_trecho['midx'] == i]
        tmini_in = tmini[tmini['ismain']==1] # Trecho principal
        tmini_out = tmini.drop(tmini_in.index) # Afluentes

        roi_df_trecho.loc[
            tmini.index, 'catid'] = tr['cotrecho']
        mtrecs.loc[ # Reasign downstream cats
            mtrecs['nutrjus'].isin(tmini_in['cotrecho']), 'nutrjus'] = tr['cotrecho']

        # Compute larger afl
        if any(tmini_out.index):
            for j, trin in tmini_out.iterrows():
                if any(tmini_out['cotrecho']==trin['nutrjus']):
                    tmini_out.loc[j, 'nucomptrec'] = trin['nucomptrec'] + tmini_out.loc[
                    tmini_out['cotrecho']==trin['nutrjus'], 'nucomptrec'].iloc[0]

                    tmini_out.loc[j, 'nodestino'] = tmini_out.loc[
                    tmini_out['cotrecho']==trin['nutrjus'], 'nodestino'].iloc[0]

            mtrecs.loc[i, 'nucompafl'] = tmini_out['nucomptrec'].max()
            mtrecs.loc[i,
                   ['noorigemafl', 'nodestinoafl']] = tmini_out.loc[
                       tmini_out['nucomptrec']==tmini_out['nucomptrec'].max(),
                       ['noorigem', 'nodestino']].iloc[0].tolist()
        else:
            mtrecs.loc[i, 'nucompafl'] = 2 * (
                mpols.loc[i, 'nuareacont'] / np.pi) ** (1/2)

    #mtrecs.loc[np.isnan(mtrecs['noorigemafl']), ['noorigemafl', 'nodestinoafl']] = 0
    #mtrecs[['noorigemafl', 'nodestinoafl']] = mtrecs[['noorigemafl', 'nodestinoafl']].astype('Int32')
    bho_trecs = roi_df_trecho[['cotrecho',
                               'cobacia',
                               'catid',
                               'midx',
                               'noorigem',
                               'nodestino',
                               'cocursodag',
                               'nucomptrec',
                               'nuareacont',
                               'nuareamont',
                               'nutrjus',
                               'geometry']]


    return mtrecs, mpols, bho_trecs


# =============================================================================
# DF operations to get river slope in trechos
# =============================================================================

def s_trec(cods_trecho, cods_ponto, corigem, cdestino, ccomp):

    m = cods_trecho.merge(cods_ponto, how='left', left_on=corigem, right_on='idponto')
    m = m.merge(cods_ponto, how='left', left_on=cdestino, right_on='idponto')
    # m = m.sort_values(by='cobacia')
    # m.set_index(cods_trecho.index, inplace=True)
    diff_elev = m['elev_x']-m['elev_y']
    river_l = cods_trecho[ccomp]

    river_s = diff_elev / river_l

    return river_s

def get_slopes(mtrecs, mpols, roi_df_ponto, demfn):

    dem = gdal.Open(demfn)
    #mxz = dem.ReadAsArray()
    nodatavalue = dem.GetRasterBand(1).GetNoDataValue()

    # Compute slopes
    elevpoint = zonal_stats(
        roi_df_ponto['geometry'],
        demfn, nodata=nodatavalue, stats=['min'], all_touched=True)
    elevpoint = [d['min'] for d in elevpoint]
    elevpoint = pd.Series(elevpoint)
    roi_df_ponto['elev'] = elevpoint

    cods_ponto = roi_df_ponto[['idponto', 'elev']]
    cods_trecho = mtrecs[[ 'cobacia',
        'noorigem', 'nodestino', 'nucomptrec',
         'noorigemafl', 'nodestinoafl', 'nucompafl']]

    mtrecs['nudecltrec'] = s_trec(cods_trecho, cods_ponto, 'noorigem', 'nodestino', 'nucomptrec')
    mtrecs['nudeclafl'] = s_trec(cods_trecho, cods_ponto, 'noorigemafl', 'nodestinoafl', 'nucompafl')

    #### Afluentes que não foram computados
    afl2fill = mpols[np.isnan(mtrecs['nudeclafl'])]
    afl2fill['nucomp'] = mtrecs['nucompafl']
    elevarea = zonal_stats(
        afl2fill['geometry'],
        demfn, nodata=nodatavalue, stats=['percentile_10', 'percentile_85'],
        all_touched=False)
    afl2fill['p10'] = [d['percentile_10'] for d in elevarea]
    afl2fill['p85'] = [d['percentile_85'] for d in elevarea]
    afl2fill['diffelev'] = afl2fill['p85'] - afl2fill['p10']
    afl2fill['nudecl'] = afl2fill['diffelev'] / afl2fill['nucomp']

    mtrecs.loc[np.isnan(mtrecs['nudeclafl']), 'nudeclafl'] = afl2fill['nudecl']

    return mtrecs

# =============================================================================
# Get HRUs
# =============================================================================
def get_hrus(mpols, hrufn):

    hru_rate = zonal_stats(
                mpols['geometry'],
                hrufn, nodata=0, categorical=True, all_touched=True)
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
def write_mini(mtrecs, mpols, hru_df, georel, smin, smax, nman):
    ## Trecho
    cobacia = mtrecs['cobacia']
    cotrecho = mtrecs['cotrecho']
    sub = mtrecs['sub']
    hdr = pd.Series([0]*len(sub))
    area_unit = mpols['nuareacont']
    area_upstream = mtrecs['nuareamont']
    river_l = mtrecs['nucomptrec']
    river_s = np.minimum(np.maximum(mtrecs['nudecltrec'], smin), smax)
    afl_l = mtrecs['nucompafl']
    afl_s = np.minimum(np.maximum(mtrecs['nudeclafl'], smin), smax)
    trechojus = mtrecs['nutrjus']
    ordem = mtrecs['nustrahler']
    nmans = pd.Series([nman]*len(sub))

    a = georel['a']; b = georel['b']; c = georel['c']; d = georel['d'];
    river_b = a * area_upstream ** b
    river_h = c * area_upstream ** d

    infos_df = pd.DataFrame({
            'CatID': cobacia,
            'Mini': cotrecho.map('{:.0f}'.format),
            'Xcen': mpols.centroid.x.map('{:.5f}'.format),
            'Ycen': mpols.centroid.y.map('{:.5f}'.format),
            'Sub': sub,
            'Area_(km2)': area_unit.map('{:.5f}'.format),
            'AreaM_(km2)': area_upstream.map('{:.5f}'.format),
            'Ltr_(km)': river_l.map('{:.5f}'.format),
            'Str_(m/km)': river_s.map('{:.5f}'.format),
            'Lrl_(km)': afl_l.map('{:.5f}'.format),
            'Srl_(m/km)': afl_s.map('{:.5f}'.format),
            'MiniJus': trechojus.map('{:.0f}'.format),
            'Ordem': ordem.map('{:.0f}'.format),
            'Hdr': hdr.map('{:.0f}'.format),
            'Width': river_b.map('{:.2f}'.format),
            'Depth': river_h.map('{:.2f}'.format),
            'NMAN': nmans.map('{:.4f}'.format),
            })

    mini_df = pd.concat([infos_df, hru_df.applymap('{:.5f}'.format)], axis=1)
    mini_df = gpd.GeoDataFrame(mini_df, geometry=mpols.geometry, crs=mpols.crs)

    mini_df.sort_values(by=['Ordem', 'CatID'], ascending=[True, False], inplace=True)

    # Change minijus to ordered 1:N
    mini_df.index = np.arange(1,len(cotrecho)+1)
    minijus = []
    for _, row in mini_df.iterrows():
        jus = mini_df.loc[mini_df['Mini']==row['MiniJus']].index.values
        minijus.extend(jus)
    minijus.extend([-1])

    mini_df['CatID'] = mini_df['Mini']
    mini_df['Mini'] = mini_df.index.values
    mini_df['MiniJus'] = minijus

    # Write mini / por algum motivo o .to_string adiciona 2 espacos entre os valores, por isso subtrair por 2
    f5 = '{:>3}'.format
    f8 = '{:>6}'.format
    f10 = '{:>8}'.format
    f15 = '{:>13}'.format

    mini_txt = mini_df.drop(['geometry'], axis=1).to_string(index=False, formatters = {
            'CatID': '{:>7}'.format,
            'Mini': '{:>7}'.format,
            'Xcen': f15,
            'Ycen': f15,
            'Sub': '{:>7}'.format,
            'Area_(km2)': f15,
            'AreaM_(km2)': f15,
            'Ltr_(km)': f15,
            'Str_(m/km)': f15,
            'Lrl_(km)': f15,
            'Srl_(m/km)': f15,
            'MiniJus': f8,
            'Ordem': f8,
            'Hdr': f5,
            'Width': f10,
            'Depth': f10,
            'NMAN': f10,
            'BLC_01': f10,
            'BLC_02': f10,
            'BLC_03': f10,
            'BLC_04': f10,
            'BLC_05': f10,
            'BLC_06': f10,
            'BLC_07': f10,
            'BLC_08': f10
            })

    return mini_df, mini_txt

# =============================================================================
# Compute HAND
# =============================================================================
# Funcao que identifica cell jusante
def flow_ds(rowL, colL, fdr):

    dir1 = fdr[rowL,colL] == 1
    dir2 = fdr[rowL,colL] == 2
    dir3 = fdr[rowL,colL] == 4
    dir4 = fdr[rowL,colL] == 8
    dir5 = fdr[rowL,colL] == 16
    dir6 = fdr[rowL,colL] == 32
    dir7 = fdr[rowL,colL] == 64
    dir8 = fdr[rowL,colL] == 128

    rowL[dir1] = rowL[dir1]
    colL[dir1] = colL[dir1] + 1

    rowL[dir2] = rowL[dir2] + 1
    colL[dir2] = colL[dir2] + 1

    rowL[dir3] = rowL[dir3] + 1
    colL[dir3] = colL[dir3]

    rowL[dir4] = rowL[dir4] + 1
    colL[dir4] = colL[dir4] - 1

    rowL[dir5] = rowL[dir5]
    colL[dir5] = colL[dir5] - 1

    rowL[dir6] = rowL[dir6] - 1
    colL[dir6] = colL[dir6] - 1

    rowL[dir7] = rowL[dir7] - 1
    colL[dir7] = colL[dir7]

    rowL[dir8] = rowL[dir8] - 1
    colL[dir8] = colL[dir8] + 1

    return rowL, colL

def coord2pixel(raster, coordX, coordY):
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    X = ((coordX - originX) / pixelWidth).astype(int)
    Y = ((coordY - originY) / pixelHeight).astype(int)
    return X, Y

def array2raster(newRasterfn,rasterfn,array):
    with rasterio.open(rasterfn) as src:
        meta = src.meta.copy()
        meta.update(compress='lzw')

    meta['dtype'] = array.dtype
    with rasterio.open(newRasterfn, 'w+', **meta) as out:
        out.write_band(1, array)

def _select_surround_ravel(i, shape): # Copied from pysheds
    """
    Select the eight indices surrounding a flattened index.
    """
    offset = shape[1]
    return np.array([i + 0 - offset,
                     i + 1 - offset,
                     i + 1 + 0,
                     i + 1 + offset,
                     i + 0 + offset,
                     i - 1 + offset,
                     i - 1 + 0,
                     i - 1 - offset]).T

def compute_hand(dem, fdr, mask, nodata_out, dirmap=(64, 128, 1, 2, 4, 8, 16, 32)): # Copied from pysheds


    # dirleft, dirright, dirtop, dirbottom = self._pop_rim(fdir, nodata=nodata_in_fdir)
    # maskleft, maskright, masktop, maskbottom = self._pop_rim(mask, nodata=0)
    mask = mask.ravel()
    r_dirmap = np.array(dirmap)[[4, 5, 6, 7, 0, 1, 2, 3]].tolist()
    source = np.flatnonzero(mask)
    hand = -np.ones(fdr.size, dtype=np.int)
    hand[source] = source
    for _ in range(fdr.size):
        selection = _select_surround_ravel(source, fdr.shape)
        ix = (fdr.flat[selection] == r_dirmap) & (hand.flat[selection] < 0)
        # TODO: Not optimized (a lot of copying here)
        parent = np.tile(source, (len(dirmap), 1)).T[ix]
        child = selection[ix]
        if not child.size:
            break
        hand[child] = hand[parent]
        source = child
    hand = hand.reshape(dem.shape)
    # if not return_index:
    hand = np.where(hand != -1, dem - dem.flat[hand], nodata_out)

    return hand


def get_hand(mtrecs, roi_df_ponto, demfn, fdrfn):

    rasfdr = gdal.Open(fdrfn)
    points = roi_df_ponto[roi_df_ponto['idponto'].isin(mtrecs['noorigem'])].geometry

    Xcoord = points.x.to_numpy()
    Ycoord = points.y.to_numpy()
    X, Y = coord2pixel(rasfdr, Xcoord, Ycoord)

    ncols, nrows = rasfdr.RasterXSize, rasfdr.RasterYSize
    str_map = np.zeros((nrows, ncols))
    str_map[(Y, X)] = 1

    fdr = rasfdr.ReadAsArray()
    # Follow fdir to fill values
    for row in range(nrows):
        for col in range(ncols):

            if str_map[row, col] == 1:
                rowL = np.array(row)
                colL = np.array(col)

                while str_map[rowL,colL] == 1:

                    rowL, colL = flow_ds(rowL, colL, fdr)

                    if ((str_map[rowL, colL] == 1)):# or
                        # (rowL not in range(10, nrows-10)) or
                        # (colL not in range(10, ncols-10))):
                        break
                    else:
                        str_map[rowL, colL] = 1

    # array2raster(folder+'str_mask.tif', fdrfn, str_map)
    rasdem = gdal.Open(demfn)
    nodata = rasdem.GetRasterBand(1).GetNoDataValue()
    dem = rasdem.ReadAsArray()

    hand = compute_hand(dem, fdr, str_map, nodata)

    return hand

from numpy import matlib

def raster_areagrid(rasterfn):

    with rasterio.open(rasterfn) as testif:
            gt = testif.transform
            pix_width = gt[0]
            ulX = gt[2]
            ulY = gt[5]
            rows = testif.height
            cols = testif.width
            lrX = ulX + gt[0] * cols
            lrY = ulY + gt[4] * rows

    lats = np.linspace(ulY,lrY,rows+1)

    a = 6378137
    b = 6356752.3142

    # Degrees to radians
    lats = lats * np.pi/180

    # Intermediate vars
    e = np.sqrt(1-(b/a)**2)
    sinlats = np.sin(lats)
    zm = 1 - e * sinlats
    zp = 1 + e * sinlats

    # Distance between meridians
    #        q = np.diff(longs)/360
    q = pix_width/360

    # Compute areas for each latitude in square km
    areas_to_equator = np.pi * b**2 * ((2*np.arctanh(e*sinlats) / (2*e) + sinlats / (zp*zm))) / 10**6
    areas_between_lats = np.diff(areas_to_equator)
    areas_cells = np.abs(areas_between_lats) * q

    areagrid = np.transpose(matlib.repmat(areas_cells,cols,1))

    return areagrid


# =============================================================================
# # Write cota-area
# =============================================================================
def cota_area(mpols, mtrecs, demfn, handfn):

    with rasterio.open(handfn) as src:
        handarr = src.read(1)
        nodata = src.nodata
        gt = src.transform


    #areagrid = raster_areagrid(handfn)

    handarr[handarr==nodata] = np.nan
    handarr[handarr>100] = np.nan
    handclass = np.copy(handarr)
    for i in np.arange(100, 0, -1):
        handclass[handarr<=i] = i


    flood_count = zonal_stats(
            mpols['geometry'],
            handclass, affine=gt, nodata=np.nan, categorical=True)
    flood_count = pd.DataFrame(flood_count).sort_index(axis=1)
    flood_count = flood_count.cumsum(axis=1).fillna(method='ffill', axis=1)

    mini_count = zonal_stats(
            mpols['geometry'],
            demfn, nodata=np.nan, stats='count')
    mini_count = pd.Series(pd.DataFrame(mini_count)['count'])

    mini_area = mpols['nuareacont']

    fa = flood_count.divide(mini_count/mini_area, axis=0)

    z0 = zonal_stats(
            mtrecs['geometry'],
            demfn, all_touched=True, stats='median')
    z0 = pd.Series(pd.DataFrame(z0)['median'])

    # Organize dataframe to correspond to MINI order
    mini_flood = fa.copy()
    mini_flood['z0'] = z0
    mini_flood[['ordem', 'cobacia']] = mtrecs[['nustrahler', 'cobacia']]
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

















# =============================================================================
# GEE functions
# =============================================================================

# =============================================================================
# Coleta dados de elevação no GEE
# =============================================================================
def get_elevations_gee(roi_fc_ponto, roi_fc_area, download_dem_map=0):

    # Carrega o DEM SRTM30 para area de interesse
    #dem = ee.Image("USGS/SRTMGL1_003").select('elevation')
    #dem = ee.Image("CGIAR/SRTM90_V4")
    dem = ee.Image("JAXA/ALOS/AW3D30/V2_2").select('AVE_DSM')
    #dem = ee.Image("WWF/HydroSHEDS/03CONDEM").select('b1')

    task_config = {
            'image': dem,
            'description': 'DEM',
            'folder': 'Iguacu',
            'scale': 30,
            'region': roi_fc_area.geometry(),
            'maxPixels': 1e9
            }
    task = ee.batch.Export.image.toDrive(**task_config)
    if download_dem_map == 1:
        task.start()

    # Coleta elevações nos pontos de entrada e saída das BHOs
    roi_elev_ponto = dem.reduceRegions(roi_fc_ponto, ee.Reducer.min(), 30)
    roi_elev_ponto = roi_elev_ponto.filter(ee.Filter.gt('min', -999))
    id_pts = roi_elev_ponto.aggregate_array('idponto').getInfo()
    pts_elev = roi_elev_ponto.aggregate_array('min').getInfo()

    s_elev = pd.Series(pts_elev, index=id_pts)

    # Coleta estatísticas das elevações nas bhos
    roi_elev_area = dem.reduceRegions(roi_fc_area, ee.Reducer.percentile([15, 85]), 30)
    p15 = np.array(roi_elev_area.aggregate_array('p15').getInfo())
    p85 = np.array(roi_elev_area.aggregate_array('p85').getInfo())
    delev_afl = p85 - p15

    return s_elev, delev_afl

def get_slopes_gee(roi_df_trecho, roi_df_ponto):

    cods_ponto = roi_df_ponto[['idponto', 'elev']]
    cods_trecho = roi_df_trecho[['noorigem', 'nodestino', 'cobacia']]

    m = cods_trecho.merge(cods_ponto, left_on='noorigem', right_on='idponto')
    m = m.merge(cods_ponto, left_on='nodestino', right_on='idponto')
    m = m.sort_values(by='cobacia')
    m.set_index(cods_trecho.index, inplace=True)
    diff_elev = m['elev_x']-m['elev_y']
    river_l = roi_df_trecho['nucomptrec']

    river_s = diff_elev / river_l

    afl_l = 2 * (roi_df_trecho['nuareacont']/np.pi) ** (1/2)
    afl_s = roi_df_trecho['delevafl'] / afl_l

    return pd.concat([river_s, afl_l, afl_s], axis=1)
    # return river_s

# =============================================================================
# Coleta dados de cobertura do solo no GEE
# =============================================================================
#    3 - Forest
#    4 - Savanna
#    5 - Mangrove
#    9 - Forest Plantation
#    10 - Non-Forest Natural
#    11 - Wetland
#    12 - Grassland
#    13 - Other Non-Forest Natural
#    15 - Pasture
#    19 - Perennial Crop
#    20 - Semi-Perennial Crop
#    21 - Mosaic Agriculture and Plasture
#    23 - Beach and Dune
#    24 - Urban
#    25 - Other Non-Vegetated
#    29 - Rocky Outcrop
#    30 - Minning
#    31 - Aquaculture
#    32 - Salt Flat
#    33 - Water Body
# HRU1 = Forest / HRU2 = Savanna / HRU3 = Farmland
# HRU4 = Wetland / HRU5 = Semi-permeable / HRU6 = Water

def get_hrus_gee(roi_fc_area, roi_df_area, lc_year, nc = 6, download_lc_map=0, download_hru_map=0):

    lc_map = (ee.Image('projects/mapbiomas-workspace/public/collection4_1/mapbiomas_collection41_integration_v1')
            .select('classification_' + str(lc_year))
            #.clip(roi_fc_area)
            )
    task_config = {
            'image': lc_map,
            'description': 'Mapbiomas_iguacu_' + str(lc_year),
            'folder': 'MAPBIOMAS\\41',
            'scale': 30,
            'region': roi_fc_area.geometry(),
            'maxPixels': 1e9
            }
    task = ee.batch.Export.image.toDrive(**task_config)
    if download_lc_map == 1:
        task.start()

    map_classes = [3, 4, 5, 9, 10, 11, 12, 13, 15, 19, 20, 21, 23, 24, 25, 29, 30, 31, 32, 33]
    hru_classes = [1, 2, 1, 1,  2,  4,  2,  2,  3,  3,  3,  3,  5,  5,  5,  5,  5,  4,  5,  6]

    hru_map = lc_map.remap(map_classes, hru_classes)

    task_config = {
            'image': hru_map,
            'description': 'HRU_iguacu_' + str(lc_year),
            'folder': folder,
            'scale': 30,
            'region': roi_fc_area.geometry(),
            'maxPixels': 1e9
            }
    task = ee.batch.Export.image.toDrive(**task_config)
    if download_hru_map == 1:
        task.start()

    hru_df = pd.DataFrame(index=roi_df_area.index)
    hru_list = list(range(1, nc+1))
    np = ee.Array(hru_map.reduceRegions(roi_fc_area, ee.Reducer.count(), 30)
                    .aggregate_array('count'))
    for hru in hru_list:
        mask = hru_map.updateMask(hru_map.eq(hru))
        hru_df['BLC_' + '{:0>2d}'.format(hru)] = (
                ee.Array(mask.reduceRegions(roi_fc_area, ee.Reducer.count(), 30)
                .aggregate_array('count')).divide(np).multiply(100).getInfo()
                )

    return hru_df

# =============================================================================
# Cota-Area
# =============================================================================
def cota_area_gee(roi_df_trecho, roi_fc_area, roi_fc_trecho, download_hand_map=0, fa=1000):

    # Get z0 for each catch by stretch mean elevation
    dem = ee.Image("USGS/SRTMGL1_003").clip(roi_fc_area)
#    river_mask = roi_fc_trecho.reduceToImage(["cotrecho"], ee.Reducer.first())
#    dem_river = dem.updateMask(river_mask)
    z0 = dem.reduceRegions(roi_fc_trecho, ee.Reducer.median(), 30).aggregate_array('median').getInfo()
    z0 = np.round(z0).astype(int)

    # Get flooded area in each catch for each z (1 to 100) using HAND (fa=100, 1000 or 5000)
    if fa not in [100, 1000, 5000]:
        print('Please provide a valid FA: 100, 1000 or 5000')
        return

    hand = ee.Image("users/gena/GlobalHAND/30m/hand-"+str(fa)).clip(roi_fc_area)

    task_config = {
            'image': hand,
            'description': 'HAND_iguacu_'+str(fa),
            'folder': 'BHO\\',
            'scale': 30,
            'region': roi_fc_area.geometry(),
            'maxPixels': 1e9
            }
    task = ee.batch.Export.image.toDrive(**task_config)
    if download_hand_map == 1:
        task.start()

    # Get data in GEE
    flood_count = (hand.reduceRegions(roi_fc_area, ee.Reducer.fixedHistogram(0, 100, 100), 30)
                    .aggregate_array('histogram').getInfo())
    mini_count = (hand.reduceRegions(roi_fc_area, ee.Reducer.count(), 30)
                    .aggregate_array('count').getInfo())
    mini_area = (ee.Image.pixelArea().divide(1000000).clip(roi_fc_area)
                    .reduceRegions(roi_fc_area, ee.Reducer.sum(), 30)
                    .aggregate_array('sum').getInfo())

    # Operations to get flooded area in km2
    flood_count_df = pd.DataFrame()
    for i in flood_count:
        i = pd.DataFrame(i)
        flood_count_df = flood_count_df.append(i[1])
    flood_count_df = flood_count_df.cumsum(axis=1)
    flood_count_df.reset_index(drop=True, inplace=True)

    fa_df = flood_count_df.divide(pd.Series(mini_count)/pd.Series(mini_area), axis=0)

    # Organize dataframe to correspond to MINI order
    mini_flood = fa_df.copy()
    mini_flood['z0'] = z0
    mini_flood[['ordem', 'cobacia']] = roi_df_trecho[['nustrahler', 'cobacia']]
    mini_flood.sort_values(by=['ordem', 'cobacia'], ascending=[True, False], inplace=True)
    mini_flood.drop(['ordem', 'cobacia'], axis=1, inplace=True)
    mini_flood.index = np.arange(1, len(mini_flood)+1)

    # Write Cota-Area
    ca_df = pd.DataFrame(columns = ['mini', 'z0', 'zfp', 'afp'])
    for mini in mini_flood.iterrows():
        mini=mini[1]
        afp = mini.drop(['z0']).astype(float)
        zi = pd.Series(mini['z0']).repeat(len(afp)).astype(int)
        zf = (afp.index + zi + 1).astype(int)
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
