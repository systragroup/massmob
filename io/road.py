import pandas as pd
from shapely.ops import transform

def _reverse_geom(geom):
    def _reverse(x, y, z=None):
        if z:
            return x[::-1], y[::-1], z[::-1]
        return x[::-1], y[::-1]
    return transform(_reverse, geom)

def split_quenedi_rlinks(road_links, oneway='0'):
    if 'oneway' not in road_links.columns:
        print('no column oneway. do not split')
        return
    links_r = road_links[road_links['oneway']==oneway].copy()
    if len(links_r) == 0:
        print('all oneway, nothing to split')
        return
    # apply _r features to the normal non r features
    r_cols = [col for col in links_r.columns if col.endswith('_r')]
    cols = [col[:-2] for col in r_cols]
    for col, r_col in zip(cols, r_cols):
        links_r[col] = links_r[r_col]
    # reindex with _r 
    links_r.index = links_r.index.astype(str) + '_r'
    # reverse links (a=>b, b=>a)
    links_r = links_r.rename(columns={'a': 'b', 'b': 'a'})
    links_r['geometry'] = links_r['geometry'].apply(lambda g: _reverse_geom(g))
    road_links = pd.concat([road_links, links_r])
    return road_links


# TODO: add OSM import from perimeter directly