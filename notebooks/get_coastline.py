
def get_coastline( ds_lon_vec, ds_lat_vec,
                   buf = 0.05,    # some buffer around AOI
                   offset = None,   # lon/lat offset to shift boundary with (in case of misalignment)
                   do_plot = False,
                   save_dir = './ancillary_data/GSHHS_shp',
                   url = r"https://www.ngdc.noaa.gov/mgg/shorelines/data/gshhg/latest/gshhg-shp-2.3.7.zip", 
                   extracted_res = 'f',   # i,h,f for intermediate, high, full -- all contained within the downloaded zip file!
                   verbose = False ):
    
    if verbose: print('Importing Python libraries ...')
    import os, requests, zipfile, io
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import Polygon
    import rasterio, rasterio.features
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from shapely.affinity import translate
    
    ### Misc.
    if verbose: print('Getting coastline data + creating land mask ...')
    cmp = LinearSegmentedColormap.from_list("cmp", ["gainsboro", "gainsboro"])   # "dummy" colormap for greyed out land pixels
    
    dlon = np.abs(np.median(np.diff(ds_lon_vec)))
    dlat = np.abs(np.median(np.diff(ds_lat_vec)))
    tmp = [ dlon, 0, np.min(ds_lon_vec)-dlon/2, 
            0, dlat, np.min(ds_lat_vec)-dlat/2 ]
    ds_affine = rasterio.Affine(*tmp)        # [xres, 0, lon, 0, yres, lat]
    
    bbox = [min(ds_lon_vec), max(ds_lon_vec), min(ds_lat_vec), max(ds_lat_vec)]
    ds_sizes = ( len(ds_lat_vec), len(ds_lon_vec) )      # (lat_size, lon_size)
                   
    ### Shape file
    shp_file = f'{save_dir}/{extracted_res}/GSHHS_{extracted_res}_L1.shp'   # file of coastline data, whole world, intermediate resolution
    shx_file = f'{save_dir}/{extracted_res}/GSHHS_{extracted_res}_L1.shx'

    if not os.path.isfile(shp_file) and not os.path.isfile(shx_file):
        if verbose: print('  Downloading .shp file from noaa.gov URL ...')
        request = requests.get(url)
        file = zipfile.ZipFile(io.BytesIO(request.content))

        file.extract(f'GSHHS_shp/{extracted_res}/GSHHS_{extracted_res}_L1.shp', path='./ancillary_data')
        file.extract(f'GSHHS_shp/{extracted_res}/GSHHS_{extracted_res}_L1.shx', path='./ancillary_data');

    bb = ( bbox[0]-buf, bbox[2]-buf, bbox[1]+buf, bbox[3]+buf )    # AOI bounding box

    ### All polygons over AOI
    if verbose: print('  Reading .shp file polygons using GPD ...')
    shp_poly = gpd.read_file( shp_file, bbox=bb )
    
    ### Fix mis-alignment between Landsat data and coastline...
    if offset is not None:
        def translate_geometry(geom, offset):
            return translate(geom, xoff=offset[0], yoff=offset[1])

        shp_poly['geometry'] = shp_poly['geometry'].apply(translate_geometry, args=(offset,))    
    
    ### Extract vector points for AOI
    if verbose: print('  Clipping .shp polygons to given ROI ...')
    polygon = Polygon([ (bb[0], bb[1]), (bb[0], bb[3]), (bb[2], bb[3]), 
                        (bb[2], bb[1]), (bb[0], bb[1]) ])
    shp_poly = gpd.clip(shp_poly, polygon)

    ### Land mask
    if verbose: print('  Rasterising features to create land mask ...')
    land_mask = rasterio.features.rasterize( ((feature['geometry'], 1) for feature in shp_poly.iterfeatures()),
                                               out_shape = (ds_sizes[0],ds_sizes[1]), transform = ds_affine )

    ### Convert the mask (numpy array) to an Xarray DataArray
    if verbose: print('  Convert land mask to Xarray ...')
    if ds_lat_vec[0]>ds_lat_vec[-1]:
        ds_lat_vec = np.array(list(reversed(ds_lat_vec)))
    land_mask = xr.DataArray(land_mask, coords=(ds_lat_vec, ds_lon_vec)).rename({'dim_0':'latitude','dim_1':'longitude'})
    land_mask = land_mask.where(land_mask==1)   # replace 0's with NaN's

    ### Plot
    if do_plot:
        if verbose: print('  Plotting results ...')
        fig = plt.figure(figsize=(8,8))
        land_mask.plot(add_colorbar=False, cmap=cmp);
        shp_poly.boundary.plot(ax=fig.axes[0], color='black', linewidth=1);
    
    return land_mask, shp_poly
    