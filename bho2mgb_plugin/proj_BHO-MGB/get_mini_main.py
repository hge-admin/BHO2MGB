# -*- coding: utf-8 -*-
"""
Created on Fri May  1 18:42:39 2020

@author: Rafael Barbedo Fontana
"""

import time
# import get_mini_functions

t_start = time.time()
# =============================================================================
# Inputs
# =============================================================================
version = '250'
folderbho = 'D:\\OneDrive\\PRH\Brasil_data\\BHO_2017_v_01_05_'+version+'\\'

# Save files
savemini = 0 # give 1 to save files
folder = "Iguacu/" # Folder where files will be saved

# lista de coordenadas das subbacias (ordenadas de jusante para montante**)
coords_list = [(-51.73, -25.98),(-50.388, -25.880)] # Iguacu Curso MGB
# coords_list = [(-36.4, -10.5)] # S Francisco

# Relacoes geomorfologicas
smin = 0.01
smax = 10000
nman = 0.030
georel = {'a': 0.89, 'b': 0.52, 'c': 0.05, 'd': 0.44}

# =============================================================================
# Etapa 1: Carrega os shapes de BHO e filtra subbacias
# =============================================================================
demfn = folder + 'dem_raw.tif'
fdrfn = folder + 'fdr.tif'
handfn = folder + 'hand_bho_250.tif'

# Carrega os shapes de BHO
bho_files = bho_getfiles(version, folderbho)
print('Loading BHO files...')
df_bho_area = carrega_bho(bho_files['area'], demfn)
df_bho_trecho = carrega_bho(bho_files['trecho'], demfn)
df_bho_ponto = carrega_bho(bho_files['ponto'], demfn)

df_bho_area.set_index('dra_pk', inplace=True)
df_bho_trecho.set_index('drn_pk', inplace=True)
df_bho_ponto.set_index('drp_pk', inplace=True)

# Carrega os shapes de BHO no GEE
# fc_bho_area = ee.FeatureCollection('projects/ee-rbfontana/assets/geoft_bho_2017_'+version+'_area_drenagem')
# fc_bho_trecho = ee.FeatureCollection('projects/ee-rbfontana/assets/geoft_bho_2017_'+version+'_trecho_drenagem')
# fc_bho_ponto = ee.FeatureCollection('projects/ee-rbfontana/assets/geoft_bho_2017_'+version+'_ponto_drenagem')

# Encontra cobacia de subbacias das coordenadas
#     Se o metodo for por nunivotto desconsiderar e usar código do nunivotto
cods = coords_in_bho(coords_list, df_bho_area)

# Coleta gdf nas áreas de interesse com atributo de subbacia
print('... Defining subbasins in region of interest')
roi_df_area, roi_df_trecho, roi_df_ponto, lista_cobacias, lista_pontos = roi_define(
        df_bho_area, df_bho_trecho, df_bho_ponto, cods)
del df_bho_area, df_bho_trecho, df_bho_ponto

# roi_fc_area = fc_bho_area.filter(ee.Filter.inList('cobacia', lista_cobacias)).sort('cobacia')
# roi_fc_trecho = fc_bho_trecho.filter(ee.Filter.inList('cobacia', lista_cobacias)).sort('cobacia')
# roi_fc_ponto = fc_bho_ponto.filter(ee.Filter.inList('idponto', lista_pontos)).sort('idponto')

# =============================================================================
# Etapa 2: Modifica BHO para atender requisitos de area e comprimentos de trecho
# =============================================================================
print('... Computing geometries')
# Aggregate set: uparea_min = 30; lmin = 6
uparea_min = 0
lmin = 0
mtrecs, mpols, bho_trecs = bho2mini(roi_df_trecho, roi_df_area, uparea_min, lmin)

# =============================================================================
# Etapa 3: Coleta declividades nos trechos BHO
# =============================================================================
print('... Computing slopes on river stretches')
mtrecs = get_slopes(mtrecs, mpols, roi_df_ponto, demfn)

# roi_df_area.to_file(folder + 'bho_area_'+version+'.shp')
# roi_df_trecho.to_file(folder + 'bho_trecho_'+version+'.shp')
# mtrecs.to_file(folder+'trechos_250.shp')
# mpols.to_file(folder+'pols_250.shp')

# =============================================================================
# Etapa 4: Coleta as HRUs
# =============================================================================
# Coleta as HRUs
print('... Collecting Hydrological Response Units')
hrufn = folder + 'HRU_iguacu_2010.tif'
hru_df = get_hrus(mpols, hrufn)

# =============================================================================
# Etapa 4: Escreve o mini com as informações obtidas e parâmetros definidos
# =============================================================================

# Escreve o mini.gtp e o mini.shp
print('... Writing MINI.gtp and MINI.shp files')
mini_gdf, mini_txt = write_mini(mtrecs, mpols, hru_df, georel, smin, smax, nman)

t_end = time.time(); print('Writed MINI.gtp successfully\nTime to process: ' + str((t_end-t_start)/60))

# =============================================================================
# Etapa 4: Escreve o arquivo de Cota-Area (somente para rodar modulo inercial)
# =============================================================================
# Compute hand by BHO
# handmap = get_hand(mtrecs, roi_df_ponto, demfn, fdrfn)
# array2raster('hand_bho_250.tif', demfn, handmap)

# Escreve o cota-area
print('... Writing COTA_AREA.flp')
t_start = time.time()
ca_txt = cota_area(mpols, mtrecs, demfn, handfn)
t_end = time.time(); print('Writed COTA_AREA.flp successfully\nTime to process: ' + str((t_end-t_start)/60) + ' min')

# =============================================================================
# Save output files
# =============================================================================

if savemini == 1:
    mini_gdf.to_file(folder + "mini_bho_"+version+".shp")

    with open(folder + "mini_bho_"+version+".gtp", "w") as text_file:
        print(mini_txt, file=text_file)
    text_file.close()

    with open(folder + "cota_area_bho_"+version+".flp", "w") as text_file:
        print(ca_txt, file=text_file)
    text_file.close()
    #roi_df_trecho.to_file(folder + "bho_"+version+"_trecho_iguacu.shp")
#







# =============================================================================
# Etapa 3: Coleta declividades
# =============================================================================
# # Carrega os shapes de BHO no GEE
# fc_bho_area = ee.FeatureCollection('projects/ee-rbfontana/assets/geoft_bho_2017_'+version+'_area_drenagem')
# fc_bho_trecho = ee.FeatureCollection('projects/ee-rbfontana/assets/geoft_bho_2017_'+version+'_trecho_drenagem')
# fc_bho_ponto = ee.FeatureCollection('projects/ee-rbfontana/assets/geoft_bho_2017_'+version+'_ponto_drenagem')

# # Filtra os shapes para a area de interesse
# roi_fc_area = fc_bho_area.filter(ee.Filter.inList('cobacia', lista_cobacias)).sort('cobacia')
# roi_fc_trecho = fc_bho_trecho.filter(ee.Filter.inList('cobacia', lista_cobacias)).sort('cobacia')
# roi_fc_ponto = fc_bho_ponto.filter(ee.Filter.inList('idponto', lista_pontos)).sort('idponto')

# ## AS
# #roi_fc_area = fc_bho_area
# #roi_fc_trecho = fc_bho_trecho
# #roi_fc_ponto = fc_bho_ponto

# # Coleta as elevações nos pontos mon-jus dos trechos
# print('... Collecting elevation data from Google Earth Engine platform')
# # Uncomment function to use
# roi_df_ponto['elev'], roi_df_trecho['delevafl'] = get_elevations(roi_fc_ponto, roi_fc_area, download_dem_map=0)
# #roi_df_ponto['elev'] = get_elevations(roi_fc_ponto, roi_fc_area, download_dem_map=0)

# # Calcula a declividade a partir das elevações dos pontos
# print('... Computing slopes on river stretches')
# roi_df_trecho[['nudecltrec', 'nucompafl', 'nudeclafl']] = get_slopes(roi_df_trecho, roi_df_ponto)
