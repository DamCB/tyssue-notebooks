import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')

import matplotlib.pylab as plt

from tyssue.core.sheet import Sheet
from tyssue import config

from tyssue.core.generation import create_anchors

from tyssue.geometry.sheet_geometry import SheetGeometry as geom
from tyssue.dynamics.apoptosis_model import SheetApoptosisModel as model
from tyssue.dynamics.sheet_vertex_model import SheetModel as basemodel
from tyssue.solvers.sheet_vertex_solver import Solver as solver
from tyssue.core.objects import get_opposite
from tyssue.draw.plt_draw import sheet_view
from tyssue.io import hdf5
from tyssue.behaviors.sheet_events import SheetEvents
from tyssue.behaviors.behaviors import apoptosis_time_table
import os
import logging
logger = logging.Logger('event_log')

min_settings = {
    'minimize': {
        'options': {
            'disp': False,
            'ftol': 1e-8,
            'gtol': 1e-8},
        }
    }


def leg_joint_view(sheet, coords=['z', 'x', 'y']):

    x, y, z = coords
    datasets = {}

    datasets['face'] = sheet.face_df.sort_values(z)
    datasets['vert'] = sheet.vert_df.sort_values(z)
    edge_z = 0.5 * (sheet.upcast_srce(sheet.vert_df[z]) +
                    sheet.upcast_trgt(sheet.vert_df[z]))
    datasets['edge'] = sheet.edge_df.copy()
    datasets['edge'][z] = edge_z
    datasets['edge'] = datasets['edge'].sort_values(z)

    tmp_sheet = Sheet('tmp', datasets,
                      sheet.specs)
    tmp_sheet.reset_index()
    cmap = plt.cm.get_cmap('viridis')

    e_depth = (tmp_sheet.edge_df[z] -
               tmp_sheet.edge_df[z].min()) / tmp_sheet.edge_df[z].ptp()
    depth_cmap = cmap(e_depth)
    draw_specs = {
        'vert': {'visible': False},
        'edge': {'color': depth_cmap}
        }

    fig, ax = sheet_view(tmp_sheet, coords[:2], **draw_specs)
    ax.set_xlim(-80, 80)
    ax.set_ylim(-50, 50)
    ax.set_axis_bgcolor('#404040')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.set_size_inches((16, 9))
    fig.set_frameon(False)
    fig.set_clip_box(ax.bbox)
    return fig, ax


def apopto_pdf(zed, theta, z0=0.,
               width_apopto=1.5, amp=0.6):
    p = (np.exp(-(zed - z0)**2 / width_apopto**2) *
         (1 - amp*(np.cos(theta/2)**2)))
    return p


def get_apopto_cells(sheet):

    np.random.seed(42)
    sheet.face_df['theta'] = np.arctan2(sheet.face_df['x'],
                                        sheet.face_df['y'])

    p_apopto = apopto_pdf(sheet.face_df['z'], sheet.face_df['theta'])
    rand_field = np.random.random(sheet.face_df.shape[0])
    apopto_cells = p_apopto[p_apopto > rand_field].index
    apopto_cells = np.array([
        c for c in sheet.face_df.loc[apopto_cells].sort_values('x').index
        if sheet.face_df.loc[c, 'is_alive']])
    return apopto_cells


def get_time_table(sheet, apopto_cells,
                   events):

    time_tables = []
    for strat_t, cell in enumerate(apopto_cells):

        times, time_table = apoptosis_time_table(sheet, cell,
                                                 events, start_t=strat_t)
        time_tables.append(time_table)
    time_table = pd.concat(time_tables).sort_index()
    times = time_table.index.levels[0]
    return times, time_table


def local_active(sheet, face):

    f_verts = sheet.edge_df[sheet.edge_df['face'] == face]['srce']
    sheet.vert_df.is_active = 0
    sheet.vert_df.is_active.loc[f_verts] = 1


def time_step(face_events, events,
              sheet, geom, model, dirname):
    for face, evts in face_events.iterrows():
        if np.isnan(face):
            continue
        for event_name, event_arg in evts.dropna().items():
            if ((not sheet.face_df.loc[face, 'is_alive']) or
                np.isnan(sheet.face_df.loc[face, 'is_alive'])):
                logger.info(
                    'skipped: face: {}, event: {}'.format(face, event_name))
                continue
            events[event_name](face, event_arg)
            logger.info('done: face: {}, event: {}'.format(face, event_name))
    res = solver.find_energy_min(sheet, geom, model, **min_settings)


def run_sim(sheet, apopto_cells,
            geom, model, dirname):
    events = SheetEvents(sheet, model, geom).events
    times, time_table = get_time_table(sheet, apopto_cells,
                                       events)
    event_logfile = os.path.join(dirname, 'events.log')
    hdlr = logging.FileHandler(event_logfile)
    hdlr.setLevel('INFO')
    logger.addHandler(hdlr)

    for t in times:
        face_events = time_table.loc[t]
        time_step(face_events, events,
                  sheet, geom, model, dirname)
        fig, ax = leg_joint_view(sheet)
        if t % 10 == 0:
            # relax boundary cells
            sheet.vert_df.is_active = 1 - sheet.vert_df.is_active
            res = solver.find_energy_min(sheet, geom, model, **min_settings)
            sheet.vert_df.is_active = 1 - sheet.vert_df.is_active

        figname = os.path.join(
            dirname, 'fold_formation_{:03d}.png'.format(t))
        plt.savefig(figname, bbox_inches='tight')
        plt.close(fig)
    logger.removeHandler(hdlr)


def single_sim(args):
    l, g, dirname, nb_dir = args
    print('Parsing' + dirname, nb_dir)

    data_file = os.path.join(nb_dir, '../data/hf5/before_apoptosis.hf5')
    datasets = hdf5.load_datasets(data_file)
    with open(os.path.join(nb_dir, 'specs.json'), 'r') as sp_file:
        specs = json.load(sp_file)
    sheet2 = Sheet('fold', datasets, specs)
    res = solver.find_energy_min(sheet2, geom, model,
                                 **min_settings)
    print('starting {}'.format(dirname))
    try:
        os.mkdir(dirname)
    except IOError:
        pass
    settings = {
        'shrink_steps': 10,
        'rad_tension': l,
        'contractile_increase': g,
        'contract_span': 3
        }
    apopto_cells = get_apopto_cells(sheet2)
    sheet2.settings['apoptosis'] = settings
    run_sim(sheet2, apopto_cells,
            geom, model, dirname)

    print('{} done'.format(dirname))
    print('~~~~~~~~~~~~~~~~~~~~~\n')
    return args
