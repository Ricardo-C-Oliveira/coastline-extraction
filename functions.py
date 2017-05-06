import rasterio
import rasterio.tools.mask
import fiona
import numpy as np
import os
import requests

def download_scene(scene_ids_list, bands_list, output_folder):
    '''
    Download Landsat8 scenes from Amazon Web Services
    :param scene_ids_list:
    :param bands_list:
    :param output_folder:
    :return: None
    '''

    # TODO: infer the path and from the input scene id
    base_url = 'http://landsat-pds.s3.amazonaws.com/L8/217/076/'

    print 'Starting to download scenes...'

    for scene in scene_ids_list:
        for band in bands_list:
            scene_url = base_url + scene + '/' + scene + '_' + band + '.TIF'
            scene_local_filename = output_folder + scene + '_' + band + '.TIF'
            scene_url = scene_url.rstrip()
            r = requests.get(scene_url, stream=True)
            with open(scene_local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print 'Scene', scene_local_filename, 'downloaded...'

    print 'All scenes downloaded!'
    return

def open_shapefile(shapepath):
    '''
    Opens a shapefile returning its geometry features.
    :param shapepath:
    :return:
    '''
    with fiona.open(shapepath, "r") as shapefile:
        features = [feature["geometry"] for feature in shapefile]

    return features

def calculate_ndwi(band3_raster_path, band5_raster_path, roi_mask):
    '''
    Calculates the NDWI of band3_raster_path and band5_raster_path. Writes a new raster on the out_folder_path
    :param band3_raster_path:
    :param band5_raster_path:
    :param out_folder_path:
    :param roi_mask:
    :return:
    '''
    out_name = os.path.basename(band3_raster_path)[:-8] + '_ndwi'

    print 'calculating', out_name

    with rasterio.open(band3_raster_path) as raster:
        band3, out_transform = rasterio.tools.mask.mask(raster, roi_mask, crop=True)
    with rasterio.open(band5_raster_path) as raster:
        band5, out_transform = rasterio.tools.mask.mask(raster, roi_mask, crop=True)

    # also returns new meta from the cropped output
    out_meta = raster.meta.copy()
    out_meta.update({"driver": "GTiff",
                 "height": band5.shape[1],
                 "width": band5.shape[2],
                 "transform": out_transform,
                     'dtype': 'float32'})

    # Calculate NDWI
    NDWI_array = (band3.astype('float32') - band5.astype('float32')) / (band3.astype('float32') + band5.astype('float32'))

    return NDWI_array.astype('float32'), out_meta

def cluster_KMeans(in_raster, n_clusters):
    '''

    :param input_raster_path:
    :param out_raster_path:
    :param n_clusters:
    :return:
    '''
    from sklearn.cluster import KMeans

    print 'Running KMeans.'


    raster_data = in_raster
    shape = raster_data.shape
    samples = np.column_stack([raster_data.flatten(), raster_data.flatten()])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(samples)
    result_kmeans = kmeans.reshape(shape).astype('float32')

    print 'Finished running KMenans.'

    return result_kmeans

def cluster_AgglomerativeClustering(in_raster, n_clusters, connectivity=False):
    '''

    :param in_raster:
    :param n_clusters:
    :param connectivity:
    :return:
    '''
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.image import grid_to_graph


    raster_data = in_raster
    shape = raster_data.shape
    samples = np.column_stack([raster_data.flatten(), raster_data.flatten()])

    if connectivity:
        print 'Running Agglomerative Clustering with Connectivity option.'
        connectivity = grid_to_graph(shape[0], shape[1])
        clf = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity)
        result_agg_cluster = clf.fit_predict(samples)
    else:
        print 'Running Agglomerative Clustering without Connectivity option.'
        clf = AgglomerativeClustering(n_clusters=n_clusters)
        result_agg_cluster = clf.fit_predict(samples)

    result_agg_cluster = result_agg_cluster.reshape(shape).astype('float32')

    print 'Finished running Agglomorative Clustering.   '

    return result_agg_cluster

def extract_contours(in_raster_obj, out_geojson, contour_const):
    '''

    :param in_raster:
    :param out_geojson:
    :param contour_const:
    :return:
    '''

    from skimage import measure
    from affine import Affine
    from pyproj import Proj, transform
    from shapely.geometry import Polygon, LineString
    import geojson

    raster_data = in_raster_obj.read(1)
    T0 = in_raster_obj.affine
    p1 = Proj(in_raster_obj.crs)
    T1 = T0 * Affine.translation(0.5, 0.5)
    rc2en = lambda r, c: (c, r) * T1
    p2 = Proj(proj='latlong',datum='WGS84')

    contours = measure.find_contours(raster_data, contour_const)

    projected_coords = []
    for c in contours[0]:
        coords = rc2en(c[0], c[1])
        projected_coords.append(transform(p1, p2, coords[0], coords[1]))

    line = LineString(projected_coords)
    with open(out_geojson, 'w') as f:
        f.write(geojson.dumps(line))
