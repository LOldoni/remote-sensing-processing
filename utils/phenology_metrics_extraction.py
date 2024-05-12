import os
import math
import json
import datetime
from typing import Union, List

import fiona
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import geopandas as gpd

import ee

# ee.Authenticate()
ee.Initialize()


def get_collection(start_date: str, end_date: str, roi: Union[ee.geometry.Geometry, ee.FeatureCollection], 
                   collection: str='COPERNICUS/S2_SR_HARMONIZAD'):
    """ Filtra uma ee.ImageCollection.

    Args:
        start_date (str): data de inicio do filtro (YYYY-MM-DD)
        end_date (str): data final do filtro (YYYY-MM-DD)
        roi (ee.geometry, ee.FeatureCollection): a geometria para se intersectar
        collection (str): o dataset desejado (https://developers.google.com/earth-engine/datasets)

    Returns:
        Uma ee.ImageCollection
    """
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    filtered_collection = ee.ImageCollection(collection) \
        .filterBounds(roi) \
        .filterDate(start, end)
    return filtered_collection


def calc_ndvi(image: ee.Image):
    """ Calcula o NDVI.

    Args:
        image: ee.Image

    Returns:
        Adiciona a imagem de NDVI na ee.Image
    """
    ndvi = image.normalizedDifference(['B8A', 'B4']).rename('NDVI')
    return image.addBands(ndvi)


def calc_evi(image: ee.Image, nir: str='B8A', red: str='B4', blue: str='B2', scale=10000):
    
    evi = image.expression(
    '2.5 * ((NIR-RED) / (NIR + 6 * RED - 7.5* BLUE +1))', {
        'NIR':image.select(nir).divide(scale),
        'RED':image.select(red).divide(scale),
        'BLUE':image.select(blue).divide(scale)
    }).rename('EVI')
    return image.addBands(evi)


def cloud_mask_qa(image: ee.Image):
    """ Aplica a S2 QA mask.

    Args:
        image(ee.Image): ee.Image

    Returns:
        Atualiza a máscara de nuvens na ee.Image
    """
    cloudShadowBitMask = (1 << 10)
    cloudsBitMask = (1 << 11)
    qa = image.select('QA60')
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(qa.bitwiseAnd(cloudsBitMask).eq(0))
    return image.updateMask(mask)


def cloud_mask_probability(image: ee.Image, max_cloud_prob: float=40):
    """ Aplica a S2 cloud probability mask.

    Args:
        image (ee.Image): ee.Image
        max_cloud_prob: O limiar de probabilidade de nuvens

    Returns:
        Atualiza a máscara de nuvens na ee.Image
    """
    clouds = ee.Image(image.get('cloud_mask')).select('probability')
    isNotCloud = clouds.lt(max_cloud_prob)
    return image.updateMask(isNotCloud)


def mask_edges(image: ee.Image):
    """ Algumas vezes a máscara para as bandas de 10m não excluem os pixels ruins nas bordas da imagem.
    Sendo necessária aplicar as máscara de 20m e 60m também.
    Referência: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_CLOUD_PROBABILITY#description

    Args:
        image (ee.Image): ee.Image a ser aplicado a máscara

    Returns:
        Atualiza a máscara de nuvens na ee.Image

    """
    return image.updateMask(image.select('B8A').mask().updateMask(image.select('B9').mask()))


def create_reduce_region_function(geometry: ee.geometry.Geometry,
                                  reducer: ee.Reducer=ee.Reducer.mean(),
                                  scale:float=20,
                                  crs: str='EPSG:4326',
                                  bestEffort: bool=True,
                                  maxPixels: float=1e13,
                                  tileScale: int=4):
    def reduce_region_function(img):
        """Aplica o método ee.Image.reduceRegion().
        Referência: https://developers.google.com/earth-engine/tutorials/community/time-series-visualization-with-altair
        Args:
          img:
            An ee.Image to reduce to a statistic by region.

        Returns:
          An ee.Feature that contains properties representing the image region
          reduction results per band and the image timestamp formatted as
          milliseconds from Unix epoch (included to enable time series plotting).
        """

        stat = img.reduceRegion(
            reducer=reducer,
            geometry=geometry,
            scale=scale,
            crs=crs,
            bestEffort=bestEffort,
            maxPixels=maxPixels,
            tileScale=tileScale)

        return ee.Feature(geometry, stat).set({'millis': img.date().millis()})

    return reduce_region_function


def fc_to_dict(fc: ee.FeatureCollection):
    """ Transfere as propriedade da feature para um dicionário.
    Referência: https://developers.google.com/earth-engine/tutorials/community/time-series-visualization-with-altair

    Args:
        fc (ee.FeatureCollection): ee.FeatureCollection

    Returns:
        Um dicionário
    """
    prop_names = fc.first().propertyNames()
    prop_lists = fc.reduceColumns(
        reducer=ee.Reducer.toList().repeat(prop_names.size()),
        selectors=prop_names).get('list')

    return ee.Dictionary.fromLists(prop_names, prop_lists)


def process_s2(start_date: str, end_date: str, polygon: ee.geometry.Geometry, vi: str='NDVI'):
    
    s2_sr = get_collection(start_date, end_date, polygon, collection='COPERNICUS/S2_SR_HARMONIZED')
    # Aplica a máscara de nuvens e calcula o NDVI
    s2_sr = s2_sr.map(calc_ndvi).map(calc_evi)
    s2_cloud_prob = get_collection(start_date, end_date, polygon, collection='COPERNICUS/S2_CLOUD_PROBABILITY')
    s2S_sr_with_cloud_mask = ee.Join.saveFirst('cloud_mask').apply(
        primary=s2_sr.map(mask_edges).map(cloud_mask_qa),
        secondary=s2_cloud_prob,
        condition=ee.Filter.equals(leftField='system:index', rightField='system:index'))
    s2_cloud_free = ee.ImageCollection(s2S_sr_with_cloud_mask).map(cloud_mask_probability).select(vi)

    return s2_cloud_free


def extract_time_serie(s2_collection: ee.ImageCollection, roi: ee.geometry.Geometry, scale: float, vi: str='NDVI'):
    # Cria uma função de redução a partir da ROI utilizando o redutor por média
    reduce_fuction = create_reduce_region_function(
        geometry=roi, reducer=ee.Reducer.mean(), scale=scale)

    # Extrai os valores de NDVI da série temporal de imagens para a ROI usando a função de redução pela média dos pixels
    # roi_stat = ee.FeatureCollection(s2_collection.map(reduce_fuction)).filter(
    #     ee.Filter.notNull(s2_collection.first().bandNames()))
    # roi_dict = fc_to_dict(roi_stat).getInfo()
    # df_stats = pd.DataFrame(roi_dict)
    # Cria um pandas dataframe como os valores de NDVI da série temporal

    roi_stat = ee.FeatureCollection(s2_collection.map(reduce_fuction))
    temp = roi_stat.getInfo()
    df_stats = pd.DataFrame.from_dict(temp['features'])
    df_stats['millis'] = df_stats.properties.apply(lambda x: x['millis'])

    df_stats['Timestamp'] = pd.to_datetime(df_stats['millis'], unit='ms')
    df_stats[vi] = df_stats.properties.apply(lambda x: x[vi] if vi in x else None)

    df_stats['tile'] = df_stats['id'].apply(lambda x: x.split('_')[-1])
    # df_stats['tile'] = df_stats['system:index'].apply(lambda x: x.split('_')[-1])

    return df_stats.loc[:, ['Timestamp', 'tile', vi]]


def dbl_logistic_model(t: np.array, p0: float, p1: float, p2: float, 
                       p3: float, p4: float, p5: float):
    """A double logistic model, as in Sobrino and Juliean,
    or Zhang et al"""
    return p1 + (p0 - p1) * (1. / (1 + np.exp(p2 * (t - p3))) + \
                        1. / (1 + np.exp(-p4 * (t - p5))) - 1)


def asymmetric_dbl_sigmoid_model(t: np.array, Vb: float, Va: float, p: float, 
                                 Di: float, q: float, Dd: float):
    """A double logistic model, as in Zhong et al 2016"""
    return Vb + 0.5 * Va * (np.tanh(p * (t - Di)) - np.tanh(q * (t - Dd)))


if __name__ == '__main__':

    # Set the start and end dates, and the vegetation index
    start_date = '2022-09-15'
    end_date = '2023-04-01'
    vegetation_index = 'EVI'
    interval = 1

    # define the point coordinates 
    x, y = -53.442137, -25.040362
    x, y = -53.421706, -25.041176
    point = ee.Geometry.Point(x, y)
    roi = point.buffer(10)

    # extract the time serie from the point with a buffer of 10 m
    s2_cloud_free = process_s2(start_date, end_date, roi, vi=vegetation_index)
    df = extract_time_serie(s2_collection=s2_cloud_free, roi=roi, scale=10, vi=vegetation_index)

    # resample the time series to a equal interval with by max value and interpolate 
    df.index = df.Timestamp
    vi_time_serie = (df.loc[:, vegetation_index].resample(f'{interval}d').max()
                     .interpolate(method="time", limit_direction='both'))
    xdata  = np.array(range(vi_time_serie.shape[0]))
    ydata = np.array(vi_time_serie)

    # initial guess for [Vb, Va, p, Di, q, Dd]
    p0 = [0.2, 0.6, 0.05, 50/interval, 0.05, 130/interval]

    # lower and upper bounds for [Vb, Va, p, Di, q, Dd]
    bounds =([0.0, 0.2, -np.inf, 0, 0, 0], 
             [0.5, 0.8, np.inf, xdata.shape[0], 0.4, xdata.shape[0]])

    # fit the model
    popt, pcov = opt.curve_fit(asymmetric_dbl_sigmoid_model, xdata=xdata, ydata=ydata, 
                               p0=p0, bounds=bounds, method='trf', maxfev=10000)
    if True in np.isinf(pcov):
        raise RuntimeError("Covariance of the parameters could not be estimated")

    Vb, Va, p, Di, q, Dd = popt

    # apply the parameters 
    vi_fitted = asymmetric_dbl_sigmoid_model(xdata, *popt)

    # reference: Zhong, Gong and Biging (2012)
    D1 = Di + np.divide(1, p) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)
    D2 = Di - np.divide(1, p) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)
    D3 = Dd + np.divide(1, p) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)
    D4 = Dd - np.divide(1, p) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)

    # Plot the time serie
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 5))
    # df.plot(y=vegetation_index, ax=ax, marker='.', lw=0, label='Raw VI')
    ax.plot(df.index, df[vegetation_index], marker='.', lw=0, label='Raw VI')
    ax.plot(vi_time_serie.index, vi_fitted, label=f'Fitted VI')
    ax.set_ylim(0, 1)
    ax.set_ylabel(vegetation_index)

    colors = ['m', 'g', 'yellow', 'c', 'orange', 'b']
    labels = ['D1', 'Di', 'D2', 'D3', 'Dd', 'D4']
    for i, d in enumerate([D1, Di, D2, D3, Dd, D4]):
        ax.axvline(vi_time_serie.index[int(d)], 0, vi_fitted[int(d)], color=colors[i], 
                   label=f'{labels[i]}: {str(vi_time_serie.index[int(d)].date())}', ls='--')

    ax.legend()