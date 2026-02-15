import sys
import math
from datetime import datetime
import numpy as np
import scipy.ndimage 
from scipy.ndimage import gaussian_filter, zoom 

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QComboBox, QDoubleSpinBox, 
                             QPushButton, QTextEdit, QGroupBox, QTabWidget,
                             QDateEdit, QFormLayout, QCheckBox, QColorDialog, 
                             QFileDialog, QSpinBox, QSlider)
from PyQt5.QtCore import Qt, QDate
from PyQt5.QtGui import QColor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr
from siphon.catalog import TDSCatalog
from xarray.backends import NetCDF4DataStore

# ==========================================
# BAZA DANYCH ZMIENNYCH GFS
# ==========================================
GFS_DATA_TREE = {
    "1. Temperatura i Wilgotność": {
        "Temperatura (2m)": "Temperature_height_above_ground",
        "Temperatura (Powierzchnia)": "Temperature_surface",
        "Temperatura odczuwalna": "Apparent_temperature_height_above_ground",
        "Temperatura maksymalna": "Maximum_temperature_height_above_ground_Mixed_intervals_Maximum",
        "Temperatura minimalna": "Minimum_temperature_height_above_ground_Mixed_intervals_Minimum",
        "Punkt rosy (2m)": "Dewpoint_temperature_height_above_ground",
        "Wilgotność względna (2m)": "Relative_humidity_height_above_ground",
        "Wilgotność właściwa (2m)": "Specific_humidity_height_above_ground",
        "Poziom zamarzania (izoterma 0°C)": "Geopotential_height_zeroDegC_isotherm",
        "Temperatura (Tropopauza)": "Temperature_tropopause",
        "Temperatura woda w glebie (Głębokość)": "Soil_temperature_depth_below_surface_layer",
    },
    "2. Opady i Śnieg": {
        "Opad deszczu (Rate)": "Precipitation_rate_surface",
        "Opad całkowity (Akumulacja)": "Total_precipitation_surface_Mixed_intervals_Accumulation",
        "Opad konwekcyjny (Rate)": "Convective_precipitation_rate_surface",
        "Kategoryczny Deszcz (Tak/Nie)": "Categorical_Rain_surface",
        "Kategoryczny Śnieg (Tak/Nie)": "Categorical_Snow_surface",
        "Kategoryczny Deszcz marznący": "Categorical_Freezing_Rain_surface",
        "Woda opadowa (Precipitable Water)": "Precipitable_water_entire_atmosphere_single_layer",
        "Głębokość śniegu": "Snow_depth_surface",
        "Woda w śniegu (Ekwiwalent)": "Water_equivalent_of_accumulated_snow_depth_surface",
        "Pokrywa lodowa": "Ice_cover_surface",
        "Procent opadów zamarzniętych": "Per_cent_frozen_precipitation_surface"
    },
    "3. Chmury i Widoczność": {
        "Zachmurzenie całkowite (%)": "Total_cloud_cover_entire_atmosphere",
        "Zachmurzenie Niskie": "Low_cloud_cover_low_cloud",
        "Zachmurzenie Średnie": "Medium_cloud_cover_middle_cloud",
        "Zachmurzenie Wysokie": "High_cloud_cover_high_cloud",
        "Zachmurzenie Konwekcyjne": "Total_cloud_cover_convective_cloud",
        "Widoczność (Powierzchnia)": "Visibility_surface",
        "Podstawa chmur konwekcyjnych (Ciśnienie)": "Pressure_convective_cloud_bottom",
        "Wierzchołek chmur konwekcyjnych (Ciśnienie)": "Pressure_convective_cloud_top",
        "Sufit chmur (Geopotencjał)": "Geopotential_height_cloud_ceiling",
        "Albedo (Powierzchnia)": "Albedo_surface_Mixed_intervals_Average"
    },
    "4. Wiatr i Dynamika": {
        "Wiatr U (10m)": "u-component_of_wind_height_above_ground",
        "Wiatr V (10m)": "v-component_of_wind_height_above_ground",
        "Porywy wiatru (Gust)": "Wind_speed_gust_surface",
        "Wiatr U (Izobaryczny)": "u-component_of_wind_isobaric",
        "Wiatr V (Izobaryczny)": "v-component_of_wind_isobaric",
        "Wiatr U (Max Wind Level)": "u-component_of_wind_maximum_wind",
        "Wiatr V (Max Wind Level)": "v-component_of_wind_maximum_wind",
        "Prędkość pionowa (Geometryczna)": "Vertical_velocity_geometric_isobaric",
        "Prędkość pionowa (Ciśnieniowa)": "Vertical_velocity_pressure_isobaric",
        "Zawirowanie absolutne (Vorticity)": "Absolute_vorticity_isobaric"
    },
    "5. Ciśnienie i Geopotencjał": {
        "Ciśnienie (MSL - Poziom Morza)": "Pressure_reduced_to_MSL_msl",
        "Ciśnienie (Powierzchnia)": "Pressure_surface",
        "Wysokość Geopotencjalna (Izobaryczna)": "Geopotential_height_isobaric",
        "Wysokość Geopotencjalna (Powierzchnia)": "Geopotential_height_surface",
        "Wysokość Geopotencjalna (Tropopauza)": "Geopotential_height_tropopause",
        "Wysokość Geopotencjalna (Max Wind)": "Geopotential_height_maximum_wind"
    },
    "6. Stabilność Atmosfery (Burze)": {
        "CAPE (Powierzchnia)": "Convective_available_potential_energy_surface",
        "CIN (Hamowanie konwekcji)": "Convective_inhibition_surface",
        "Lifted Index (Powierzchnia)": "Surface_Lifted_Index_surface",
        "Best 4-layer Lifted Index": "Best_4_layer_Lifted_Index_surface",
        "Storm Relative Helicity": "Storm_relative_helicity_height_above_ground_layer",
        "U-Storm Motion": "U-Component_Storm_Motion_height_above_ground_layer",
        "V-Storm Motion": "V-Component_Storm_Motion_height_above_ground_layer",
        "Haines Index (Pożary)": "Haines_index_surface"
    },
    "7. Gleba i Powierzchnia": {
        "Wilgotność gleby (Objętościowa)": "Volumetric_Soil_Moisture_Content_depth_below_surface_layer",
        "Wilgotność gleby (Płynna)": "Liquid_Volumetric_Soil_Moisture_non_Frozen_depth_below_surface_layer",
        "Parowanie potencjalne": "Potential_Evaporation_Rate_surface",
        "Spływ powierzchniowy (Runoff)": "Water_runoff_surface_Mixed_intervals_Accumulation",
        "Typ gleby": "Soil_type_surface",
        "Pokrycie terenu (Land Cover)": "Land_cover_0__sea_1__land_surface",
        "Roślinność (%)": "Vegetation_surface",
        "Szorstkość terenu": "Surface_roughness_surface"
    },
    "8. Promieniowanie i Energia": {
        "Strumień ciepła jawnego": "Sensible_heat_net_flux_surface_Mixed_intervals_Average",
        "Strumień ciepła utajonego": "Latent_heat_net_flux_surface_Mixed_intervals_Average",
        "Promieniowanie krótkofalowe (W dół)": "Downward_Short-Wave_Radiation_Flux_surface_Mixed_intervals_Average",
        "Promieniowanie długofalowe (W dół)": "Downward_Long-Wave_Radp_Flux_surface_Mixed_intervals_Average",
        "Promieniowanie krótkofalowe (W górę)": "Upward_Short-Wave_Radiation_Flux_surface_Mixed_intervals_Average",
        "Czas nasłonecznienia": "Sunshine_Duration_surface"
    }
}

REGIONS_DATA = {
    "Polska": (55.0, 49.0, 14.0, 24.5),
    "Niemcy": (55.1, 47.2, 5.8, 15.1),
    "Czechy": (51.1, 48.5, 12.0, 18.9),
    "Ukraina": (52.4, 44.3, 22.1, 40.3),
    "Europa": (72.0, 30.0, -25.0, 45.0),
    "Świat": (90.0, -90.0, -180.0, 180.0),
    "Własny": (None, None, None, None)
}

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111, projection=ccrs.PlateCarree())
        super(MplCanvas, self).__init__(self.fig)

class WeatherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GFS Explorer v15.0 (Super Smooth Curves)")
        self.resize(1400, 950)
        
        self.ds = None          
        self.data_var = None    
        self.lons = None
        self.lats = None
        self.meta_unit = ""
        self.meta_name = ""
        self.meta_time = None
        self.current_extent = None

        # Zmienne stylowe
        self.style_cmap = 'jet'
        self.style_levels = 60
        self.style_contour_on = False
        self.style_contour_col = '#000000'
        self.style_contour_width = 0.7
        self.style_grid_on = True
        self.style_borders_col = '#000000'
        
        # Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # --- PANEL BOCZNY (TABS) ---
        side_container = QWidget()
        side_container.setFixedWidth(400)
        side_layout = QVBoxLayout(side_container)
        layout.addWidget(side_container)

        self.tabs = QTabWidget()
        side_layout.addWidget(self.tabs)

        # === ZAKŁADKA 1: DANE I REGION ===
        tab_data = QWidget()
        vbox_data = QVBoxLayout(tab_data)

        # 1. Parametry
        gb_var = QGroupBox("1. Wybór Parametrów")
        vbox_var = QVBoxLayout()
        vbox_var.addWidget(QLabel("Kategoria:"))
        self.combo_category = QComboBox()
        self.combo_category.addItems(GFS_DATA_TREE.keys())
        self.combo_category.currentIndexChanged.connect(self.update_variable_list)
        vbox_var.addWidget(self.combo_category)
        vbox_var.addWidget(QLabel("Zmienna:"))
        self.combo_var = QComboBox()
        self.update_variable_list()
        vbox_var.addWidget(self.combo_var)
        gb_var.setLayout(vbox_var)
        vbox_data.addWidget(gb_var)

        # 2. Data
        gb_date = QGroupBox("2. Data i Czas (UTC)")
        vbox_date = QVBoxLayout()
        hbox_datetime = QHBoxLayout()
        self.date_edit = QDateEdit(QDate.currentDate())
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setCalendarPopup(True)
        
        self.combo_time = QComboBox()
        hours = [f"{h:02d}:00" for h in range(0, 24, 3)]
        self.combo_time.addItems(hours)
        
        curr_h = datetime.utcnow().hour
        best_match = min(hours, key=lambda x: abs(int(x.split(':')[0]) - curr_h))
        self.combo_time.setCurrentText(best_match)

        hbox_datetime.addWidget(self.date_edit)
        hbox_datetime.addWidget(self.combo_time)
        vbox_date.addLayout(hbox_datetime)
        
        self.btn_latest = QPushButton("Teraz (Latest)")
        self.btn_latest.clicked.connect(self.set_latest_datetime)
        vbox_date.addWidget(self.btn_latest)

        gb_date.setLayout(vbox_date)
        vbox_data.addWidget(gb_date)

        # 3. Region
        gb_region = QGroupBox("3. Region")
        vbox_region = QVBoxLayout()
        self.combo_regions = QComboBox()
        self.combo_regions.addItems(list(REGIONS_DATA.keys())[:-1])
        self.combo_regions.addItem("Własny")
        self.combo_regions.currentIndexChanged.connect(self.on_region_changed)
        vbox_region.addWidget(self.combo_regions)
        
        grid_coords = QFormLayout()
        self.spin_north = self.create_spinbox(55.0, -90, 90)
        self.spin_south = self.create_spinbox(49.0, -90, 90)
        self.spin_west = self.create_spinbox(14.0, -180, 180)
        self.spin_east = self.create_spinbox(24.5, -180, 180)
        grid_coords.addRow("N:", self.spin_north)
        grid_coords.addRow("S:", self.spin_south)
        grid_coords.addRow("W:", self.spin_west)
        grid_coords.addRow("E:", self.spin_east)
        vbox_region.addLayout(grid_coords)
        gb_region.setLayout(vbox_region)
        vbox_data.addWidget(gb_region)

        self.btn_fetch = QPushButton("POBIERZ DANE I RYSUJ")
        self.btn_fetch.setStyleSheet("background-color: #D32F2F; color: white; font-weight: bold; padding: 10px;")
        self.btn_fetch.clicked.connect(self.fetch_gfs_data)
        vbox_data.addWidget(self.btn_fetch)

        vbox_data.addStretch()
        self.tabs.addTab(tab_data, "Dane")

        # === ZAKŁADKA 2: STYL I WYGLĄD ===
        tab_style = QWidget()
        vbox_style = QVBoxLayout(tab_style)

        # 1. Kolory Mapy
        gb_colors = QGroupBox("Kolorystyka Danych")
        form_colors = QFormLayout()
        
        self.cmb_cmap = QComboBox()
        cmaps = ['jet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                 'coolwarm', 'bwr', 'seismic', 'twilight', 'hsv', 'Spectral', 'RdYlBu_r', 'Blues', 'Greens', 'Reds', 'Greys']
        self.cmb_cmap.addItems(cmaps)
        self.cmb_cmap.currentTextChanged.connect(self.refresh_plot_local)
        
        self.spin_levels = QSpinBox()
        self.spin_levels.setRange(10, 200)
        self.spin_levels.setValue(60)
        self.spin_levels.valueChanged.connect(self.refresh_plot_local)

        form_colors.addRow("Paleta barw:", self.cmb_cmap)
        form_colors.addRow("Liczba poziomów:", self.spin_levels)
        gb_colors.setLayout(form_colors)
        vbox_style.addWidget(gb_colors)

        # === NOWE: Wygładzanie (ULEPSZONE) ===
        gb_smooth = QGroupBox("Jakość i Wygładzanie (Interpolacja)")
        vbox_smooth = QVBoxLayout()
        
        hbox_smooth_title = QHBoxLayout()
        hbox_smooth_title.addWidget(QLabel("Moc wygładzania:"))
        self.lbl_smooth_val = QLabel("0 (Wył)")
        hbox_smooth_title.addWidget(self.lbl_smooth_val)
        vbox_smooth.addLayout(hbox_smooth_title)
        
        self.slider_smooth = QSlider(Qt.Horizontal)
        self.slider_smooth.setRange(0, 10) # 0 = Wyłączone, 10 = Max
        self.slider_smooth.setValue(3)
        self.slider_smooth.setTickPosition(QSlider.TicksBelow)
        self.slider_smooth.setTickInterval(1)
        # Zdarzenie release, żeby nie zamulało przy przesuwaniu
        self.slider_smooth.sliderReleased.connect(self.refresh_plot_local)
        self.slider_smooth.valueChanged.connect(self.update_smooth_label)

        vbox_smooth.addWidget(self.slider_smooth)
        vbox_smooth.addWidget(QLabel("<i>Wyższe wartości dają gładsze linie, ale rysowanie trwa dłużej.</i>"))
        
        gb_smooth.setLayout(vbox_smooth)
        vbox_style.addWidget(gb_smooth)
        # ==========================

        # 2. Izolinie
        gb_iso = QGroupBox("Izolinie (Kontury)")
        vbox_iso = QVBoxLayout()
        self.chk_show_iso = QCheckBox("Włącz izolinie")
        self.chk_show_iso.setChecked(False)
        self.chk_show_iso.toggled.connect(self.refresh_plot_local)
        
        hbox_iso_sets = QHBoxLayout()
        self.btn_iso_color = QPushButton("Kolor")
        self.btn_iso_color.setStyleSheet(f"background-color: {self.style_contour_col}; color: white;")
        self.btn_iso_color.clicked.connect(self.pick_iso_color)
        
        self.spin_iso_width = QDoubleSpinBox()
        self.spin_iso_width.setRange(0.1, 5.0)
        self.spin_iso_width.setValue(0.7)
        self.spin_iso_width.setSingleStep(0.1)
        self.spin_iso_width.valueChanged.connect(self.refresh_plot_local)
        
        self.chk_iso_labels = QCheckBox("Etykiety liczb")
        self.chk_iso_labels.setChecked(True)
        self.chk_iso_labels.toggled.connect(self.refresh_plot_local)

        hbox_iso_sets.addWidget(self.btn_iso_color)
        hbox_iso_sets.addWidget(QLabel("Grubość:"))
        hbox_iso_sets.addWidget(self.spin_iso_width)
        
        vbox_iso.addWidget(self.chk_show_iso)
        vbox_iso.addLayout(hbox_iso_sets)
        vbox_iso.addWidget(self.chk_iso_labels)
        gb_iso.setLayout(vbox_iso)
        vbox_style.addWidget(gb_iso)

        # 3. Mapa Tła
        gb_map = QGroupBox("Elementy Mapy")
        vbox_map = QVBoxLayout()
        
        self.chk_grid_vis = QCheckBox("Siatka współrzędnych")
        self.chk_grid_vis.setChecked(True)
        self.chk_grid_vis.toggled.connect(self.refresh_plot_local)
        
        hbox_borders = QHBoxLayout()
        self.btn_border_color = QPushButton("Kolor Granic")
        self.btn_border_color.setStyleSheet("background-color: black; color: white;")
        self.btn_border_color.clicked.connect(self.pick_border_color)
        hbox_borders.addWidget(self.btn_border_color)

        vbox_map.addWidget(self.chk_grid_vis)
        vbox_map.addLayout(hbox_borders)
        gb_map.setLayout(vbox_map)
        vbox_style.addWidget(gb_map)

        self.btn_refresh_style = QPushButton("ODŚWIEŻ WYGLĄD (Bez pobierania)")
        self.btn_refresh_style.setStyleSheet("background-color: #1976D2; color: white; padding: 8px;")
        self.btn_refresh_style.clicked.connect(self.refresh_plot_local)
        vbox_style.addWidget(self.btn_refresh_style)

        vbox_style.addStretch()
        self.tabs.addTab(tab_style, "Styl i Wygląd")

        # === LOGI I ZAPIS ===
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumHeight(150)
        side_layout.addWidget(QLabel("Logi systemu:"))
        side_layout.addWidget(self.log_console)

        self.btn_save_img = QPushButton("ZAPISZ MAPĘ (JPG/PNG)")
        self.btn_save_img.setStyleSheet("background-color: #388E3C; color: white; font-weight: bold; padding: 15px;")
        self.btn_save_img.clicked.connect(self.save_map_image)
        side_layout.addWidget(self.btn_save_img)

        # --- OKNO MAPY ---
        map_layout = QVBoxLayout()
        layout.addLayout(map_layout, 1)
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        map_layout.addWidget(self.toolbar)
        map_layout.addWidget(self.canvas)
        
        self.combo_regions.setCurrentIndex(0)
        self.log("Aplikacja gotowa. Wybierz parametry i kliknij 'POBIERZ DANE'.")

    def create_spinbox(self, val, min_val, max_val):
        sb = QDoubleSpinBox()
        sb.setRange(min_val, max_val)
        sb.setValue(val)
        sb.setSingleStep(0.5)
        sb.setDecimals(2)
        return sb

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_console.append(f"[{timestamp}] {message}")
        sb = self.log_console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def update_variable_list(self):
        selected_category = self.combo_category.currentText()
        variables_dict = GFS_DATA_TREE.get(selected_category, {})
        self.combo_var.blockSignals(True)
        self.combo_var.clear()
        for friendly_name, raw_name in variables_dict.items():
            self.combo_var.addItem(friendly_name, raw_name)
        self.combo_var.blockSignals(False)

    def on_region_changed(self):
        name = self.combo_regions.currentText()
        if name == "Własny": return
        vals = REGIONS_DATA.get(name)
        if vals and vals[0] is not None:
            self.spin_north.setValue(vals[0])
            self.spin_south.setValue(vals[1])
            self.spin_west.setValue(vals[2])
            self.spin_east.setValue(vals[3])

    def set_latest_datetime(self):
        now = datetime.utcnow()
        self.date_edit.setDate(now.date())
        hours_list = [self.combo_time.itemText(i) for i in range(self.combo_time.count())]
        best_match = min(hours_list, key=lambda x: abs(int(x.split(':')[0]) - now.hour))
        self.combo_time.setCurrentText(best_match)
        self.log(f"Ustawiono czas na teraz: {now.date()} {best_match} UTC")

    def pick_iso_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.style_contour_col = color.name()
            self.btn_iso_color.setStyleSheet(f"background-color: {color.name()}; color: {'black' if color.lightness() > 128 else 'white'};")
            self.refresh_plot_local()

    def pick_border_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.style_borders_col = color.name()
            self.btn_border_color.setStyleSheet(f"background-color: {color.name()}; color: {'black' if color.lightness() > 128 else 'white'};")
            self.refresh_plot_local()

    def update_smooth_label(self, val):
        if val == 0:
            self.lbl_smooth_val.setText("0 (Wył)")
        else:
            self.lbl_smooth_val.setText(f"Poziom {val}")

    def fetch_gfs_data(self):
        self.btn_fetch.setEnabled(False)
        self.btn_fetch.setText("POBIERANIE...")
        self.log("Rozpoczynam pobieranie danych...")
        QApplication.processEvents()

        try:
            friendly_name = self.combo_var.currentText()
            var_name = self.combo_var.currentData()
            if not var_name: raise ValueError("Brak zmiennej")

            q_date = self.date_edit.date()
            time_str = self.combo_time.currentText()
            parts = time_str.split(':')
            query_time = datetime(q_date.year(), q_date.month(), q_date.day(), int(parts[0]), int(parts[1]))

            n = self.spin_north.value()
            s = self.spin_south.value()
            w = self.spin_west.value()
            e = self.spin_east.value()
            self.current_extent = [w, e, s, n]

            catalog_url = 'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/catalog.xml'
            catalog = TDSCatalog(catalog_url)
            dataset_access = catalog.datasets['Best GFS Quarter Degree Forecast Time Series']
            ncss = dataset_access.subset()
            
            query = ncss.query()
            query.variables(var_name).time(query_time)
            query.lonlat_box(north=n, south=s, east=e, west=w)
            query.accept('netcdf4')

            self.log(f"Zapytanie: {var_name} na {query_time}")
            data = ncss.get_data(query)
            self.ds = xr.open_dataset(NetCDF4DataStore(data))
            
            data_var = self.ds[var_name].squeeze()
            if len(data_var.shape) == 3: data_var = data_var[0, :, :]
            
            self.data_var = data_var
            self.lons = self.ds['longitude']
            self.lats = self.ds['latitude']
            self.meta_unit = self.ds[var_name].attrs.get('units', '?')
            self.meta_name = friendly_name
            self.meta_time = query_time

            self.log("Dane pobrane pomyślnie.")
            self.refresh_plot_local()

        except Exception as e:
            self.log(f"BŁĄD POBIERANIA: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.btn_fetch.setEnabled(True)
            self.btn_fetch.setText("POBIERZ DANE I RYSUJ")

    def refresh_plot_local(self):
        if self.data_var is None:
            self.log("Brak danych do wyświetlenia. Najpierw pobierz dane.")
            return

        # Pokazanie kursora oczekiwania, bo wygładzanie może chwilę trwać
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            self.canvas.fig.clf()
            ax = self.canvas.fig.add_subplot(111, projection=ccrs.PlateCarree())
            self.canvas.axes = ax
            
            ax.set_extent(self.current_extent, crs=ccrs.PlateCarree())

            cmap = self.cmb_cmap.currentText()
            levels = self.spin_levels.value()
            
            
            plot_lons = self.lons
            plot_lats = self.lats
            plot_data = self.data_var
            
            smooth_level = self.slider_smooth.value()
            
            if smooth_level > 0:
                # 1. Obliczamy współczynnik Zoomu (od 2x do ok 8x-10x)
                # Dla poziomu 1 -> zoom 1.5, dla 10 -> zoom 8.0
                zoom_factor = 1.0 + (smooth_level * 0.7)
                
                # 2. Obliczamy Sigma dla filtru Gaussa (żeby usunąć ostre rogi pikseli)
                # Zbyt duża sigma rozmyje dane, zbyt mała zostawi "kantyzm"
                sigma_val = smooth_level * 0.15
                
                self.log(f"Przetwarzanie: Zoom {zoom_factor:.1f}x, Rozmycie {sigma_val:.1f}...")

                # Interpolacja danych (Zoom) - to robi największą robotę w zagęszczaniu linii
                plot_data = zoom(plot_data, zoom_factor, order=3)
                plot_lons = zoom(plot_lons, zoom_factor, order=1)
                plot_lats = zoom(plot_lats, zoom_factor, order=1)
                
                # Filtr Gaussa - usuwa "schodkowanie" (aliasing) po zoomie
                plot_data = gaussian_filter(plot_data, sigma=sigma_val)
            

            # 1. Wypełnione kontury
            cf = ax.contourf(plot_lons, plot_lats, plot_data, levels, 
                             transform=ccrs.PlateCarree(), cmap=cmap)
            
            # 2. Izolinie
            if self.chk_show_iso.isChecked():
                iso_width = self.spin_iso_width.value()
                iso_col = self.style_contour_col
                
                # Redukcja liczby linii w stosunku do poziomów (żeby nie było czarno od linii)
                c_levels = max(5, int(levels / 3)) 
                
                cs = ax.contour(plot_lons, plot_lats, plot_data, c_levels, 
                                transform=ccrs.PlateCarree(), 
                                colors=iso_col, linewidths=iso_width, alpha=0.8)
                
                if self.chk_iso_labels.isChecked():
                    ax.clabel(cs, inline=True, fontsize=9, fmt='%1.1f')

            # 3. Elementy mapy
            ax.coastlines(resolution='10m', color=self.style_borders_col, linewidth=1)
            ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor=self.style_borders_col)
            
            if self.chk_grid_vis.isChecked():
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=1, color='gray', alpha=0.4, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': 8}
                gl.ylabel_style = {'size': 8}

            cbar = self.canvas.fig.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
            cbar.set_label(f"{self.meta_name} [{self.meta_unit}]")

            ax.set_title(f"GFS: {self.meta_name}\n{self.meta_time.strftime('%Y-%m-%d %H:%M UTC')}", fontsize=10)

            self.canvas.draw()
            self.log("Rysowanie zakończone.")

        except Exception as e:
            self.log(f"Błąd renderowania: {e}")
            import traceback
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()

    def save_map_image(self):
        if self.data_var is None:
            self.log("Brak mapy do zapisania.")
            return
            
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Zapisz mapę jako", "", "Images (*.jpg *.png);;All Files (*)", options=options)
        if fileName:
            try:
                self.canvas.fig.savefig(fileName, dpi=150, bbox_inches='tight')
                self.log(f"Zapisano mapę: {fileName}")
            except Exception as e:
                self.log(f"Błąd zapisu: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = WeatherApp()
    window.show()
    sys.exit(app.exec_())