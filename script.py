import rasterio
import glob
import os
import time
import numpy as np
from functions import cluster_AgglomerativeClustering, cluster_KMeans, extract_contours, calculate_ndwi, open_shapefile, download_scene

# Define the scene to download
# right now the download script has this path and row hardcoded
scenes_ids = ['LC82170762014233LGN00']

bands = ['B3', 'B5']

# Download scene
download_scene(scenes_ids, bands, '/Users/ricardooliveira/GIS/RS_Project/rs_riodejaneiro/input_data/')

# Set in and out directories
in_dir = '/Users/ricardooliveira/GIS/RS_Project/rs_riodejaneiro/input_data/'
out_dir = '/Users/ricardooliveira/GIS/RS_Project/rs_riodejaneiro/output_ndwi/'

raw_files = glob.glob(in_dir + '*')

# Open ROI shapefile
roi = open_shapefile('/Users/ricardooliveira/GIS/RS_Project/rs_riodejaneiro/shapes/study_area_utm.shp')

for idx, file in enumerate(raw_files):
    if idx % 2 == 0:
        print file, raw_files[idx + 1], '\n'
        file_name = os.path.basename(file)[:-7]
        ndwi_array, out_meta_ndwi = calculate_ndwi(file, raw_files[idx + 1], roi)

        ndwi_array = np.squeeze(ndwi_array)

        with rasterio.open('{}{}_ndwi.tif'.format(out_dir, file_name), 'w', **out_meta_ndwi) as dest:
            dest.write(ndwi_array, indexes=1)

        # Run Agglomerative Clustering with connectivity option
        with rasterio.open('{}{}_agg_clustering_with_connectivity.tif'.format(out_dir, file_name), 'w', **out_meta_ndwi) as dest:
            agg_cluster_connectivity_out = cluster_AgglomerativeClustering(ndwi_array, 2, connectivity=True)
            dest.write(agg_cluster_connectivity_out, indexes=1)

        with rasterio.open('{}{}_agg_clustering_with_connectivity.tif'.format(out_dir, file_name), 'r') as agg_cluster_connectivity_raster:
            extract_contours(agg_cluster_connectivity_raster, '{}{}_agg_clustering_with_connectivity.json'.format(out_dir, file_name), 0)

        # Run Agglomerative Clustering without connectivity option
        with rasterio.open('{}{}_agg_clustering_without_connectivity.tif'.format(out_dir, file_name), 'w', **out_meta_ndwi) as dest:
            agg_cluster_connectivity_out = cluster_AgglomerativeClustering(ndwi_array, 2)
            dest.write(agg_cluster_connectivity_out, indexes=1)

        with rasterio.open('{}{}_agg_clustering_without_connectivity.tif'.format(out_dir, file_name), 'r') as agg_cluster_connectivity_raster:
            extract_contours(agg_cluster_connectivity_raster, '{}{}_agg_clustering_without_connectivity.json'.format(out_dir, file_name), 0)

        # Run kmeans
        with rasterio.open('{}{}_cluster_kmeans_out.tif'.format(out_dir, file_name), 'w', **out_meta_ndwi) as dest:
            cluster_kmeans_out = cluster_KMeans(ndwi_array, 2)
            dest.write(cluster_kmeans_out, indexes=1)

        with rasterio.open('{}{}_cluster_kmeans_out.tif'.format(out_dir, file_name), 'r') as agg_cluster_connectivity_raster:
            extract_contours(agg_cluster_connectivity_raster, '{}{}_cluster_kmeans_out.json'.format(out_dir, file_name), 0)