from sentinelhub import BBox


def to_bbox(gdf):
    minx, miny, maxx, maxy = gdf.total_bounds
    bbox = BBox([minx, miny, maxx, maxy], crs=f'EPSG:{gdf.crs.to_epsg()}')
    return bbox
