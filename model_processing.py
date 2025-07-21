from ultralytics import YOLO
from PIL import Image, ImageDraw  # Tambahkan ImageDraw
import geopandas as gpd
from shapely.geometry import Point
import rasterio
import numpy as np
from sklearn.cluster import DBSCAN
import os
import types
from types import GeneratorType
import zipfile
import shapefile
import xml.etree.ElementTree as ET
from pyproj import Transformer
import json

Image.MAX_IMAGE_PIXELS = None  # Remove pixel limitation

def sliding_window(image, step_size, window_size):
    for y in range(0, image.height, step_size):
        for x in range(0, image.width, step_size):
            yield x, y, image.crop((x, y, x + window_size[0], y + window_size[1]))

def process_image(image_path, chm_path, chm_filter, confidence, iou, cluster_eps):
    print(f"Parameter input: chm_filter={chm_filter}, confidence={confidence}, iou={iou}, cluster_eps={cluster_eps}")

    # Load pre-trained YOLOv8 model
    model = YOLO("weights/best.pt")
    output_folder = "static/images"

    with rasterio.open(image_path) as src:
        image = Image.open(image_path)
        transform = src.transform
        crs = src.crs

    window_size = (2048, 2048)
    step_size = 2048

    detected_points = []  # Inisialisasi list kosong untuk menyimpan titik-titik yang terdeteksi

    with rasterio.open(chm_path) as chm_src:  # Membuka file CHM menggunakan rasterio
        chm_array = chm_src.read(1)  # Membaca data CHM sebagai array 2D
        height, width = chm_array.shape  # Mendapatkan tinggi dan lebar dari array CHM

        # Loop melalui setiap window yang dihasilkan oleh fungsi sliding_window
        for x, y, window in sliding_window(image, step_size, window_size):
            # Melakukan prediksi menggunakan model YOLO pada window dengan confidence threshold 0.10 dan IOU threshold 0.1
            results = model(window, conf=0.15, iou=0.1)
            # Mengubah hasil prediksi menjadi list jika hasilnya adalah generator
            results_list = list(results) if isinstance(results, types.GeneratorType) else results
            # Mendapatkan data bounding box dari hasil prediksi jika atribut 'data' ada
            boxes_data = results_list[0].boxes.data if hasattr(results_list[0].boxes, 'data') else []
            # Mengubah data bounding box menjadi list jika data tidak kosong
            boxes_list = boxes_data[:, :4].tolist() if boxes_data.nelement() else []

            # Loop melalui setiap bounding box dalam boxes_list
            for box in boxes_list:
                # Mendapatkan koordinat x1, y1, x2, y2 dari bounding box
                x1, y1, x2, y2 = box
                # Menghitung koordinat pusat bounding box dalam window
                center_x = (x1 + x2) / 2 + x
                center_y = (y1 + y2) / 2 + y
                # Mengubah koordinat pusat bounding box menjadi titik dalam sistem koordinat gambar
                point = Point(transform * (center_x, center_y))
                # Mendapatkan indeks baris dan kolom dari titik dalam array CHM
                row, col = chm_src.index(point.x, point.y)
                # Memeriksa apakah titik berada dalam batas array CHM dan nilai CHM di titik tersebut >= 2.46
                if 0 <= row < height and 0 <= col < width and chm_array[row, col] >= 2.46:
                    # Menambahkan titik yang terdeteksi ke dalam list detected_points
                    detected_points.append(point)

    print(f"Jumlah pohon yang terdeteksi: {len(detected_points)}")

    gdf = gpd.GeoDataFrame(geometry=[Point(point) for point in detected_points], crs=crs)
    final_gdf = cluster_points(gdf, eps=cluster_eps, min_samples=1)
    final_output_path = os.path.join(output_folder, "Points_shapefile.shp")
    final_gdf.to_file(final_output_path)

    # Tambahkan verifikasi
    print(f"File SHP disimpan di: {final_output_path}")
    print(f"Jumlah fitur dalam file SHP: {len(final_gdf)}")

    # Zip the shapefile
    shapefile_base = os.path.splitext(final_output_path)[0]
    zip_output_path = shapefile_base + ".zip"
    with zipfile.ZipFile(zip_output_path, 'w') as zipf:
        for ext in ['.shp', '.shx', '.dbf', '.prj']:
            file_path = shapefile_base + ext
            if os.path.exists(file_path):
                zipf.write(file_path, os.path.basename(file_path))

    # Convert SHP to OSM
    osm_output_path = shapefile_base + ".osm"
    shp_to_osm(final_output_path, osm_output_path, image_path)

    tree_count = len(final_gdf)
    
    return zip_output_path, tree_count, osm_output_path

def cluster_points(gdf, eps, min_samples):
    coords = np.array(list(gdf.geometry.apply(lambda geom: (geom.x, geom.y))))
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_
    
    # Tambahkan logging
    print(f"Jumlah titik sebelum clustering: {len(coords)}")
    print(f"Jumlah cluster yang ditemukan: {len(set(labels)) - (1 if -1 in labels else 0)}")
    
    clustered_points = [Point(coords[labels == k].mean(axis=0)) for k in set(labels) if k != -1]
    print(f"Jumlah titik setelah clustering: {len(clustered_points)}")
    
    return gpd.GeoDataFrame(geometry=clustered_points, crs=gdf.crs)

def shp_to_osm(shp_path, osm_path, image_path):
    # Membaca file SHP
    sf = shapefile.Reader(shp_path)
    
    # Tambahkan logging untuk memeriksa isi file SHP
    print(f"Jumlah record dalam SHP: {len(sf)}")
    print(f"Fields dalam SHP: {sf.fields}")
    
    fields = sf.fields[1:]  # Mengabaikan field pertama yang merupakan 'DeletionFlag'
    field_names = [field[0] for field in fields]

    # Membuka file sumber dengan rasterio untuk mendapatkan proyeksi
    with rasterio.open(image_path) as src:
        src_crs = src.crs
        transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    # Tambahkan logging untuk memeriksa konversi koordinat
    print(f"CRS sumber: {src_crs}")

    # Membuat elemen root OSM
    root = ET.Element("osm", version="0.6", generator="shp_to_osm")

    for sr in sf.shapeRecords():
        record = sr.record
        shape = sr.shape

        # Membuat elemen node untuk setiap titik dalam shape
        for point in shape.points:
            lon, lat = transformer.transform(point[0], point[1])
            print(f"Koordinat asli: {point[0]}, {point[1]}")
            print(f"Koordinat yang dikonversi: {lon}, {lat}")

            node_elem = ET.SubElement(root, "node", id=str(id(point)), lat=str(lat), lon=str(lon))

            # Menambahkan tag untuk setiap field dalam record
            for field_name, value in zip(field_names, record):
                tag_elem = ET.SubElement(node_elem, "tag", k=field_name, v=str(value))

    # Menyimpan ke file OSM
    tree = ET.ElementTree(root)
    tree.write(osm_path, encoding='utf-8', xml_declaration=True)

def convert_to_geojson(gdf):
    # Pastikan CRS diatur dengan benar
    gdf = gdf.to_crs(epsg=4326)
    
    # Konversi ke GeoJSON
    geojson = json.loads(gdf.to_json())
    
    print(f"Jumlah fitur dalam GeoJSON: {len(geojson['features'])}")
    
    return geojson