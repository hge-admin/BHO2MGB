# -*- coding: utf-8 -*-
"""
/***************************************************************************
 MGB
                                 A QGIS plugin
 A plugin.
                             -------------------
        begin                : 2021-05-10
        copyright            : (C) 2021 by Grupo de Pesquisa Hidrologia de Grande Escala
        email                : leolaipelt@gmail.com
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load MGB class from file MGB.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .MGB import MGB
    return MGB(iface)
