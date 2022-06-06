# -*- coding: utf-8 -*-
"""
/***************************************************************************
 MGB
                                 A QGIS plugin
 A plugin.
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2021-05-10
        git sha              : $Format:%H$
        copyright            : (C) 2021 by Grupo de Pesquisa Hidrologia de Grande Escala
        email                : leolaipelt@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QFileDialog, QMessageBox

# Initialize Qt resources from file resources.py
from .resources import *
# Import the code for the dialog
from .MGB_dialog import MGBDialog

import pathlib
import sys
import os.path

# def installer_func():
#     plugin_dir = os.path.dirname(os.path.realpath(__file__))

#     try:
#         import pip
#     except ImportError:
#         exec(
#             open(str(pathlib.Path(plugin_dir, 'scripts', 'get_pip.py'))).read()
#         )
#         import pip
#         # just in case the included version is old
#         pip.main(['install', '--upgrade', 'pip'])

#     sys.path.append(plugin_dir)

#     with open(os.path.join(plugin_dir,'requirements.txt'), "r") as requirements:
#         for dep in requirements.readlines():
#             dep = dep.strip().split("==")[0]
#             try:
#                 __import__(dep)
#             except ImportError as e:
#                 print("{} not available, installing".format(dep))
#                 pip.main(['install', dep])

# installer_func()

import pyproj
import time

from time import sleep

from qgis.core import (
    Qgis, QgsVectorLayer, QgsApplication, QgsMessageLog, QgsTask, QgsProject)

from .model import *
from .pysheds.grid import Grid
#from .grid import Grid
# from .affine import Affine
# from .richdem import rdarray, BreachDepressions, FillDepressions, ResolveFlats

class MGB:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'MGB_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&BHO2MGB')

        # Check if plugin was started the first time in current QGIS session
        # Must be set in initGui() to survive plugin reloads
        self.first_start = None


    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('MGB', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            # Adds plugin icon to Plugins toolbar
            self.iface.addToolBarIcon(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/MGB/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'MGB'),
            callback=self.run,
            parent=self.iface.mainWindow())

        # will be set False in run()
        self.first_start = True


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&BHO2MGB'),
                action)
            self.iface.removeToolBarIcon(action)
            



    def run(self):
        """Run method that performs all the real work"""

        #accepted vector formatters
        self.filter = "ESRI Shape Files (*.shp);;OGC GeoPackage (*.gpkg);;"

        #cleaning QtWidgets


        # Create the dialog with elements (after translation) and keep reference
        # Only create GUI ONCE in callback, so that it will only load when the plugin is started
        if self.first_start == True:
            self.first_start = False
            self.dlg = MGBDialog()

        # Open buttons
            self.dlg.input_BHO_areas.setText("")
            self.dlg.bho_open_areas.clicked.connect(self.open_BHO_areas_file)
            self.dlg.bho_open_trecs.clicked.connect(self.open_BHO_trecs_file)
            #self.dlg.bho_open_point.clicked.connect(self.open_BHO_point_file)

            self.dlg.dem_open_3.clicked.connect(self.open_DEM_file_tab1)


            #self.dlg.fdr_open.clicked.connect(self.open_DEM_file)

        # Output buttons
            self.dlg.open_output.clicked.connect(self.output_directory)

        # Checkbox inercial
            #self.dlg.inercial_checkbox.clicked.connect(self.state_checkbox_inercial)

        # Run button
            self.dlg.run_button.clicked.connect(self.run_2)

        #Step 2

            self.dlg.rd_areas_button.clicked.connect(self.open_roi_define_areas)
            self.dlg.rd_trecs_button.clicked.connect(self.open_roi_define_trecs)

            self.dlg.open_output_step2.clicked.connect(self.output_directory_table2)

            self.dlg.run_button_step2.clicked.connect(self.run_step2)


        #Step 3
            self.dlg.mtrecs_open.clicked.connect(self.open_mtrecs)
            self.dlg.mareas_open.clicked.connect(self.open_mareas)
            self.dlg.dem_open.clicked.connect(self.open_DEM_file)
            self.dlg.hru_open.clicked.connect(self.open_hru_file)
            self.dlg.open_output_2.clicked.connect(self.output_directory_table3)

            self.dlg.run_button_2.clicked.connect(self.run_step3)

        # Text Align
        #self.dlg.inputDEM.setAlignment(Qt.AlignRight)

        # show the dialog
        self.dlg.show()

        self.dlg.exec_()
        # Run the dialog event loop
        #result = self.dlg.exec_()
        # See if OK was pressed
        #if result:
            # Do something useful here - delete the line containing pass and
            # substitute with your code.
            #pass

    def run_2(self):
            globals()['task1'] = Step1('Delimitation of the watershed.', 10,
            self.dlg.output_folder.toPlainText(),
            self.dlg.input_BHO_areas.toPlainText(),
            self.dlg.input_BHO_trecs.toPlainText(),
            self.dlg.input_DEM_3.toPlainText(),
            self.dlg.table_coords,
            self.dlg.progressBar,
            self.dlg
            )

            #print('task step 2 status: ',tsk.status())
            QgsApplication.taskManager().addTask(globals()['task1'])
            #print('task step 2 status: ',tsk.status())

    def run_step2(self):
            globals()['task2'] = Step2('Aggregation.', 10,
            self.dlg.output_step2_folder.toPlainText(),
            self.dlg.input_roi_define_areas.toPlainText(),
            self.dlg.input_roi_define_trecs.toPlainText(),
            self.dlg.input_area_min.toPlainText(),
            self.dlg.input_lmin.toPlainText(),
            self.dlg.progressBar_2,
            self.dlg,
            )

            #print('task step 2 status: ',tsk.status())
            QgsApplication.taskManager().addTask(globals()['task2'])
            #print('task step 2 status: ',tsk.status())

    def run_step3(self):
            globals()['task3'] = Step3('Creating Mini.gtp and CotaArea.flp.', 10,
            self.dlg.output_folder_2.toPlainText(),
            self.dlg.input_DEM.toPlainText(),
            self.dlg.input_mtrecs.toPlainText(),
            self.dlg.input_mareas.toPlainText(),
            self.dlg.input_hru.toPlainText(),
            self.dlg.geo_a.toPlainText(),
            self.dlg.geo_b.toPlainText(),
            self.dlg.geo_c.toPlainText(),
            self.dlg.geo_d.toPlainText(),
            self.dlg.geo_smin.toPlainText(),
            self.dlg.geo_smax.toPlainText(),
            self.dlg.geo_nman.toPlainText(),
            self.dlg.progressBar_3,
            self.dlg

            )
            #print('task step 2 status: ',tsk.status())
            QgsApplication.taskManager().addTask(globals()['task3'])
            #print('task step 2 status: ',tsk.status())
    #def state_checkbox_inercial(self, int):
        #if self.dlg.inercial_checkbox.isChecked():
            #self.dlg.fdr_open.setEnabled(True)
            #self.dlg.input_Fdr.setDisabled(False)
        #else:
            #self.dlg.fdr_open.setEnabled(False)
            #self.dlg.input_Fdr.setDisabled(True)
    def open_BHO_areas_file(self):

        fileName = QFileDialog.getOpenFileName(None, 'OpenFile','',filter=self.filter)
        self.dlg.input_BHO_areas.setText(fileName[0])
        #self.dlg.input_bho_areas2.setText(fileName[0])

    def open_BHO_trecs_file(self):

        fileName = QFileDialog.getOpenFileName(None, 'OpenFile','',filter=self.filter)
        self.dlg.input_BHO_trecs.setText(fileName[0])
        #self.dlg.input_bho_trecs2.setText(fileName[0])

    def open_BHO_areas_file2(self):

        fileName = QFileDialog.getOpenFileName(None, 'OpenFile','',filter=self.filter)
        self.dlg.input_bho_areas2.setText(fileName[0])

    def open_BHO_trecs_file2(self):

        fileName = QFileDialog.getOpenFileName(None, 'OpenFile','',filter=self.filter)
        self.dlg.input_bho_trecs2.setText(fileName[0])

    def open_BHO_point_file(self):

        fileName = QFileDialog.getOpenFileName(None, 'OpenFile','',filter=self.filter)
        self.dlg.input_BHO_point.setText(fileName[0])

    def open_DEM_file(self):

        fileName = QFileDialog.getOpenFileName(None,'OpenFile','', "Raster Files (*.asc *.irst *tif)")
        self.dlg.input_DEM.setText(fileName[0])

    def open_DEM_file_tab1(self):

        fileName = QFileDialog.getOpenFileName(None,'OpenFile','', "Raster Files (*.asc *.irst *tif)")
        self.dlg.input_DEM_3.setText(fileName[0])
        self.dlg.input_DEM.setText(fileName[0])

    def open_FDR_file(self):

        fileName = QFileDialog.getOpenFileName(None,'OpenFile','', "Raster Files (*.asc *.irst *tif)")
        self.dlg.input_Fdr.setText(fileName[0])

    def open_hru_file(self):
        fileName = QFileDialog.getOpenFileName(None,'OpenFile','', "Raster Files (*.asc *.irst *tif)")
        self.dlg.input_hru.setText(fileName[0])

    def open_mtrecs(self):
        fileName =  QFileDialog.getOpenFileName(None,'OpenFile','',filter=self.filter)
        self.dlg.input_mtrecs.setText(fileName[0])

    def open_mareas(self):
        fileName =  QFileDialog.getOpenFileName(None,'OpenFile','',filter=self.filter)
        self.dlg.input_mareas.setText(fileName[0])

    def output_directory(self):

        #fileName = QFileDialog.getOpenFileName(None,'SaveFile','', "Raster Files (*.asc *.irst *tif)")
        fileName = QFileDialog.getExistingDirectory(None, 'Select a Directory')
        self.dlg.output_folder.setText(fileName)
        self.dlg.output_step2_folder.setText(fileName)
        self.dlg.output_folder_2.setText(fileName)


    def output_directory_table3(self):

        #fileName = QFileDialog.getOpenFileName(None,'SaveFile','', "Raster Files (*.asc *.irst *tif)")
        fileName = QFileDialog.getExistingDirectory(None, 'Select a Directory')
        self.dlg.output_folder_2.setText(fileName)

    def open_roi_define_areas(self):

        fileName =  QFileDialog.getOpenFileName(None,'OpenFile','',filter=self.filter)
        self.dlg.input_roi_define_areas.setText(fileName[0])

    def open_roi_define_trecs(self):

        fileName =  QFileDialog.getOpenFileName(None,'OpenFile','',filter=self.filter)
        self.dlg.input_roi_define_trecs.setText(fileName[0])


    def output_directory_table2(self):

        #fileName = QFileDialog.getOpenFileName(None,'SaveFile','', "Raster Files (*.asc *.irst *tif)")
        fileName = QFileDialog.getExistingDirectory(None, 'Select a Directory')
        self.dlg.output_step2_folder.setText(fileName)


MESSAGE_CATEGORY = 'Step1 task'

class Step1(QgsTask):
    """This shows how to subclass QgsTask"""

    def __init__(self, description, duration,directory, bho_areas, bho_trecs, dem_file_input, table_points,progressbar,dlg_widgets):

        super().__init__(description, QgsTask.CanCancel)
        self.duration = duration
        self.total = 0
        self.iterations = 0
        self.exception = None
        self.directory = directory
        self.bho_areas_file = bho_areas
        self.bho_trecs_file = bho_trecs
        self.dem_file = dem_file_input
        self.table_coord = table_points
        self.progress_value = 0
        self.progressbar = progressbar
        self.time_start = time.time()
        self.dlg = dlg_widgets
    
    def run(self):
        """Here you implement your heavy lifting. This method should
        periodically test for isCancelled() to gracefully abort.
        This method MUST return True or False
        raising exceptions will crash QGIS so we handle them internally and
        raise them in self.finished
        """

        def progress_count(init,end):
            for n in range(init,end+1,1):
                self.setProgress(n)
                self.progressbar.setValue(n)
                sleep(0.02)

            self.progress_value = end

            return

        QgsMessageLog.logMessage('Started task "{}"'.format(
            self.description()), MESSAGE_CATEGORY, Qgis.Info)
        #wait_time = self.duration / 100


        directory = os.path.join(self.directory, 'output\\')

        if not os.path.exists(directory):
            os.makedirs(directory)

        """"Change directory"""
        os.chdir(directory)

        #t_start = time.time()

        bho_areas_file = self.bho_areas_file
        bho_trecs_file = self.bho_trecs_file
        dem_file = self.dem_file

        progress_count(self.progress_value, 10)

        df_bho_areas, output_file_areas = carrega_bho(bho_areas_file, dem_file, 'bho_areas')

        progress_count(self.progress_value, 20)

        df_bho_trecs, output_file_trecs = carrega_bho(bho_trecs_file, dem_file, 'bho_trecs')

        progress_count(self.progress_value, 30)

        df_bho_areas.set_index('dra_pk', inplace=True)
        df_bho_trecs.set_index('drn_pk', inplace=True)
        #df_bho_ponto.set_index('drp_pk', inplace=True)

        list_coords = []
        for n in range(0,int(self.table_coord.rowCount())):
            try:
                list_coords.append(int(self.table_coord.item(n,0).text()))
            except:
                list_coords.append(-9999)
        cods = [str(i) for i in list_coords if i >= 0]
        cods = sorted(cods)

        #cods = float(self.table_coord.item(0,0).text())
        #print(cods)
        #lat = float(self.table_coord.item(0,1).text())

        progress_count(self.progress_value, 40)

        #coords_list = [(lon,lat)]

        #cods = coords_in_bho(coords_list, output_file_areas)
        QgsVectorFileWriter.deleteShapeFile(output_file_areas)
        QgsVectorFileWriter.deleteShapeFile(output_file_trecs)

        progress_count(self.progress_value, 50)

        roi_areas, roi_trecs = roi_define(
        df_bho_areas, df_bho_trecs, cods)
        del df_bho_areas, df_bho_trecs

        progress_count(self.progress_value, 100)

        self.result_bho_areas =  roi_areas
        self.result_bho_trecs =  roi_trecs

        # check isCanceled() to handle cancellation
        if self.isCanceled():
            return False

        return True

    def finished(self, result):
        """This method is automatically called when self.run returns.
        result is the return value from self.run.
        This function is automatically called when the task has completed (
        successfully or otherwise). You just implement finished() to do
        whatever
        follow up stuff should happen after the task is complete. finished is
        always called from the main thread, so it's safe to do GUI
        operations and raise Python exceptions here.
        """
        if result:

            #self.result_bho_areas
            Prj = QgsProject().instance() # Object for current project

            self.dlg.input_roi_define_areas.setText(self.result_bho_areas)
            self.dlg.input_roi_define_trecs.setText(self.result_bho_trecs)

            #mareas_view = QgsVectorLayer(self.result_bho_areas, 'catchs', 'ogr')
            #mtrecs_view = QgsVectorLayer(self.result_bho_trecs, 'rivers', 'ogr')

            #Prj.addMapLayers([mtrecs_view, mareas_view])

            QgsMessageLog.logMessage(
                'Task "{name}" completed\n' \
                'Time duration: {total} seconds.'.format(
                    name=self.description(),
                    total= "{:.2f}".format(time.time()-self.time_start),
                    ),
                MESSAGE_CATEGORY, Qgis.Success)
        else:
            if self.exception is None:
                QgsMessageLog.logMessage(
                    'Task "{name}" not successful but without exception ' \
                    '(probably the task was manually canceled by the '
                    'user)'.format(
                        name=self.description()),
                    MESSAGE_CATEGORY, Qgis.Warning)
            else:
                QgsMessageLog.logMessage(
                    'Task "{name}" Exception: {exception}'.format(
                        name=self.description(), exception=self.exception),
                    MESSAGE_CATEGORY, Qgis.Critical)
                raise self.exception

    def cancel(self):
        QgsMessageLog.logMessage(
            'Task "{name}" was cancelled'.format(name=self.description()),
            MESSAGE_CATEGORY, Qgis.Info)
        super().cancel()


MESSAGE_CATEGORY = 'Step 2 task.'
class Step2(QgsTask):
    """This shows how to subclass QgsTask"""

    def __init__(self, description, duration, directory, roi_areas, roi_trecs, area_min, lmin, progressbar, dlg_widgets):

        super().__init__(description, QgsTask.CanCancel)
        self.duration = duration
        self.total = 0
        self.iterations = 0
        self.exception = None
        self.directory = directory
        self.roi_areas = roi_areas
        self.roi_trecs = roi_trecs
        self.area_min = area_min
        self.lmin = lmin
        self.progressbar = progressbar
        self.progress_value = 0
        self.time_start = time.time()
        self.dlg = dlg_widgets
    
    def run(self):
        """Here you implement your heavy lifting. This method should
        periodically test for isCancelled() to gracefully abort.
        This method MUST return True or False
        raising exceptions will crash QGIS so we handle them internally and
        raise them in self.finished
        """
        def progress_count(init,end):
            for n in range(init,end+1,1):
                self.setProgress(n)
                self.progressbar.setValue(n)
                sleep(0.02)

            self.progress_value = end

            return

        QgsMessageLog.logMessage('Started task "{}"'.format(
            self.description()), MESSAGE_CATEGORY, Qgis.Info)
        #wait_time = self.duration / 100
        directory = os.path.join(self.directory, 'output\\')

        if not os.path.exists(directory):
            os.makedirs(directory)

        roi_trecs_input = self.roi_trecs
        roi_areas_input = self.roi_areas

        roi_df_trecs = load_df(roi_trecs_input)
        roi_df_areas = load_df(roi_areas_input)

        uparea_min = self.area_min
        lmin = self.lmin

        progress_count(self.progress_value, 40)

        mtrecs, mareas = bho2mini(roi_df_trecs, roi_df_areas, uparea_min, lmin)

        progress_count(self.progress_value, 100)

        self.mtrecs_out = mtrecs
        self.mareas_out = mareas

        # check isCanceled() to handle cancellation
        if self.isCanceled():
            return False

        return True

    def finished(self, result):
        """This method is automatically called when self.run returns.
        result is the return value from self.run.
        This function is automatically called when the task has completed (
        successfully or otherwise). You just implement finished() to do
        whatever
        follow up stuff should happen after the task is complete. finished is
        always called from the main thread, so it's safe to do GUI
        operations and raise Python exceptions here.
        """
        if result:

            Prj = QgsProject().instance() # Object for current project

            self.dlg.input_mareas.setText(self.mareas_out)
            self.dlg.input_mtrecs.setText(self.mtrecs_out)

            mareas_view = QgsVectorLayer(self.mareas_out,'aggregated catchments','ogr')
            mtrecs_view = QgsVectorLayer(self.mtrecs_out,'aggregated streams','ogr')
            Prj.addMapLayers([mtrecs_view, mareas_view])

            QgsMessageLog.logMessage(
                'Task "{name}" completed\n' \
                'Time duration: {total} seconds.'.format(
                    name=self.description(),
                    total= "{:.2f}".format(time.time()-self.time_start),
                    ),
                MESSAGE_CATEGORY, Qgis.Success)
        else:
            if self.exception is None:
                QgsMessageLog.logMessage(
                    'Task "{name}" not successful but without exception ' \
                    '(probably the task was manually canceled by the '
                    'user)'.format(
                        name=self.description()),
                    MESSAGE_CATEGORY, Qgis.Warning)
            else:
                QgsMessageLog.logMessage(
                    'Task "{name}" Exception: {exception}'.format(
                        name=self.description(), exception=self.exception),
                    MESSAGE_CATEGORY, Qgis.Critical)
                raise self.exception

    def cancel(self):
        QgsMessageLog.logMessage(
            'Task "{name}" was cancelled'.format(name=self.description()),
            MESSAGE_CATEGORY, Qgis.Info)
        super().cancel()

MESSAGE_CATEGORY = 'Step 3 task.'
class Step3(QgsTask):
    """This shows how to subclass QgsTask"""

    def __init__(self, description, duration, directory, input_dem, input_mtrecs, input_mareas,input_hru,
    geoa, geob, geoc, geod, geomin, geomsmax, geonman, progressbar,dlg_widgets
    ):

        super().__init__(description, QgsTask.CanCancel)
        self.duration = duration
        self.total = 0
        self.iterations = 0
        self.exception = None
        self.directory = directory
        self.input_dem = input_dem
        self.input_mtrecs = input_mtrecs
        self.input_mareas = input_mareas
        self.input_hru = input_hru
        self.geoa = geoa
        self.geob = geob
        self.geoc = geoc
        self.geod = geod
        self.geomin = geomin
        self.geomsmax = geomsmax
        self.geonman = geonman
        self.progressbar = progressbar
        self.dlg = dlg_widgets

        self.progress_value = 0
        self.time_start = time.time()
    
    def run(self):
        """Here you implement your heavy lifting. This method should
        periodically test for isCancelled() to gracefully abort.
        This method MUST return True or False
        raising exceptions will crash QGIS so we handle them internally and
        raise them in self.finished
        """
        def progress_count(init,end):
            for n in range(init,end+1,1):
                self.setProgress(n)
                self.progressbar.setValue(n)
                sleep(0.02)

            self.progress_value = end

            return

        QgsMessageLog.logMessage('Started task "{}"'.format(
            self.description()), MESSAGE_CATEGORY, Qgis.Info)
        #wait_time = self.duration / 100

        demfn = self.input_dem
        directory = os.path.join(self.directory, 'output\\')

        if not os.path.exists(directory):
            os.makedirs(directory)

        mtrecsfn = self.input_mtrecs
        mareasfn = self.input_mareas
        moutlinesfn = os.path.join(directory, 'moutlines.shp')
        
        handfn = os.path.join(directory, 'hand.tif')
        ltndfn = os.path.join(directory, 'ltnd.tif')

        progress_count(self.progress_value, 10)


        # =============================================================================
        # DEM processing
        # =============================================================================
        grid = Grid().from_raster(demfn, 'dem')
        grid.rasterize(mareasfn, out_name='basin', nodata_out=0, apply_mask=True)
        grid.clip_to('basin')
        
        pols2lines(mareasfn, moutlinesfn)
        grid.rasterize(moutlinesfn, out_name='edges')
        QgsVectorFileWriter.deleteShapeFile(moutlinesfn)
        
        grid.rasterize(mtrecsfn, out_name='streams')
        
        grid.burn_dem('dem', 'streams', out_name='burned_dem')
        
        progress_count(self.progress_value, 20)
        
        # Determine D8 flow directions from DEM
        # ----------------------
        # Fill depressions in DEM
        # pysheds method ----------------------
        # grid.fill_depressions('burned_dem', out_name='flooded_dem')
        
        # Resolve flats in DEM
        # grid.resolve_flats('flooded_dem', out_name='inflated_dem')
        
        # richdem method ----------------------

        # nodata = grid.nodata
        # gt = grid.affine
        # proj = grid.crs

        # dem = rdarray(grid.view('burned_dem'), no_data=nodata)
        # dem.geotransform = Affine.to_gdal(gt)
        # dem.projection = proj

        # BreachDepressions(dem, in_place=True)
        # FillDepressions(dem, epsilon=False, in_place=True)
        # ResolveFlats(dem, in_place=True)

        # grid.add_gridded_data(dem, 'filled_dem',
        #                       affine=gt, shape=dem.shape, crs=proj, nodata=nodata)
        
        progress_count(self.progress_value, 40)
        # Compute flow directions
        # -------------------------------------
        # Specify directional mapping
        # dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        
        # grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap)
        # grid.view('dir')
        
        grid.compute_hand_strbrn('burned_dem', 'dem', 'streams', 'edges')
        grid.compute_ltnd_strbrn('burned_dem', 'streams', 'edges')

        grid.to_raster('hand', handfn)
        grid.to_raster('ltnd', ltndfn)

        progress_count(self.progress_value, 50)
        
        # =============================================================================
        # Collect slopes and HRUs
        # =============================================================================
        mtrecs = load_df(mtrecsfn)
        mareas = load_df(mareasfn)
        
        # Collect slopes
        mtrecs['nudecltrec'] = get_slopes_main(mtrecsfn, demfn)

        nudeclafl_list, nucompafl_list = get_slopes_afl(mareasfn, handfn, ltndfn)
        mtrecs['nudeclafl'] = nudeclafl_list
        mtrecs['nucompafl'] = nucompafl_list
        
        # Collect HRUs
        hrufn = self.input_hru
        hru_df = get_hrus(mareasfn, hrufn)

        progress_count(self.progress_value, 60)
        
        # =============================================================================
        # Write MINI and COTA-AREA files
        # =============================================================================
        geo_a = float(self.geoa)
        geo_b = float(self.geob)
        geo_c = float(self.geoc)
        geo_d = float(self.geod)
        georel = {'a': geo_a, 'b': geo_b, 'c':geo_c, 'd': geo_d}

        smin = float(self.geomin)
        smax = float(self.geomsmax)
        nman = float(self.geonman)

        # Write mini.gtp e o mini.shp
        mini_df, mini_txt = write_mini(mtrecs, mareas, hru_df, georel, smin, smax, nman)

        minigtp_file = os.path.join(directory, 'MINI.gtp')
        tfile = open(minigtp_file, 'w')
        tfile.write(mini_txt)
        tfile.close()

        progress_count(self.progress_value, 75)

        ca_txt = write_cota_area(mareasfn, mtrecsfn, demfn, handfn)

        cotaarea_file = os.path.join(directory, 'COTA_AREA.flp')
        cfile = open(cotaarea_file, 'w')
        cfile.write(ca_txt)
        cfile.close()

        progress_count(self.progress_value, 90)
        
        # Write mini shapefiles        
        output_file = os.path.join(directory, 'minis_mgb.shp')
        save_df(mini_df, output_file, ogr.wkbPolygon)
        
        progress_count(self.progress_value, 95)
        
        progress_count(self.progress_value, 100)

        # check isCanceled() to handle cancellation
        if self.isCanceled():
            return False

        return True

    def finished(self, result):
        """This method is automatically called when self.run returns.
        result is the return value from self.run.
        This function is automatically called when the task has completed (
        successfully or otherwise). You just implement finished() to do
        whatever
        follow up stuff should happen after the task is complete. finished is
        always called from the main thread, so it's safe to do GUI
        operations and raise Python exceptions here.
        """
        if result:
            QgsMessageLog.logMessage(
                'Task "{name}" completed\n' \
                'Time duration: {total} seconds.'.format(
                    name=self.description(),
                    total= "{:.2f}".format(time.time()-self.time_start),
                    ),
                MESSAGE_CATEGORY, Qgis.Success)
        else:
            if self.exception is None:
                QgsMessageLog.logMessage(
                    'Task "{name}" not successful but without exception ' \
                    '(probably the task was manually canceled by the '
                    'user)'.format(
                        name=self.description()),
                    MESSAGE_CATEGORY, Qgis.Warning)
            else:
                QgsMessageLog.logMessage(
                    'Task "{name}" Exception: {exception}'.format(
                        name=self.description(), exception=self.exception),
                    MESSAGE_CATEGORY, Qgis.Critical)
                raise self.exception

    def cancel(self):
        QgsMessageLog.logMessage(
            'Task "{name}" was cancelled'.format(name=self.description()),
            MESSAGE_CATEGORY, Qgis.Info)
        super().cancel()
