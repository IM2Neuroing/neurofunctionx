"""Stimulation-parameter queries against the SQLite database.

Plain SQL over a ``db_path``; returns pandas DataFrames. No BIDS knowledge.
"""
import sqlite3

import pandas as pd


def get_intra_simulations_parameters(db_path, subject_id):
    return _get_simulations_parameters(db_path, subject_id, session="Intraoperativ")


def get_post_simulations_parameters(db_path, subject_id):
    return _get_simulations_parameters(db_path, subject_id, session="Postoperativ")


def _get_simulations_parameters(db_path, subject_id, session):
    with sqlite3.connect(db_path) as conn:
        query = """
                SELECT hemisphere, trajectory_type, position_on_trajectory, amplitude, stimulation_mode, rigidity_tremor
                FROM stimulations
                         LEFT JOIN stimulation_sessions
                                   ON stimulations.stim_session_id = stimulation_sessions.stim_session_id
                         LEFT JOIN stimulation_evaluations
                                   ON stimulations.stimulation_eval_id = stimulation_evaluations.stimulation_eval_id
                         LEFT JOIN subjects
                                   ON stimulation_sessions.subject_id = subjects.subject_id
                         LEFT JOIN occurred_side_effects
                                   ON stimulations.stimulation_id = occurred_side_effects.stimulation_id
                WHERE subjects.subject_id = ?
                  AND stimulation_sessions.session_name = ?
                """
        return pd.read_sql_query(query, conn, params=[subject_id, session])


def get_chronic_simulations_parameters(db_path, subject_id):
    with sqlite3.connect(db_path) as conn:
        query = """
                SELECT DISTINCT csp.hemisphere, csp.stimulation_mode, cs.amplitude, c.contact_number, ti.electrode_name
                FROM chronic_stim_parameters as csp
                         LEFT JOIN contacts as c
                                   ON csp.chronic_stim_parameter_id = c.chronic_stim_parameter_id
                         LEFT JOIN contact_setup as cs
                                   ON c.contact_setup_id = cs.contact_setup_id
                         LEFT JOIN targeting_info as ti
                                   ON csp.subject_id = ti.subject_id
                where csp.subject_id = ?
                  and cs.amplitude != 'NAN'
                """
        return pd.read_sql_query(query, conn, params=[subject_id])
