# ── Constants for Silica Soft Sensor ──────────────────────────────

BOUNDS = {
    '% Iron Feed':                   (42.74, 65.78),
    '% Silica Feed':                 (1.31,  33.40),
    'Starch Flow':                   (54.6,  6270.2),
    'Amina Flow':                    (242.9, 737.0),
    'Ore Pulp Flow':                 (376.8, 418.1),
    'Ore Pulp pH':                   (8.75,  10.81),
    'Flotation Column 01 Air Flow':  (175.9, 312.3),
    'Flotation Column 02 Air Flow':  (178.2, 309.9),
    'Flotation Column 03 Air Flow':  (177.2, 302.8),
    'Flotation Column 06 Air Flow':  (196.5, 355.0),
    'Flotation Column 07 Air Flow':  (199.7, 351.3),
    'Flotation Column 01 Level':     (181.9, 859.0),
    'Flotation Column 02 Level':     (224.9, 827.8),
    'Flotation Column 03 Level':     (135.2, 884.8),
    'Flotation Column 04 Level':     (165.7, 675.6),
    'Flotation Column 05 Level':     (214.7, 674.1),
    'Flotation Column 06 Level':     (203.7, 698.5),
    'Flotation Column 07 Level':     (185.1, 655.5),
    '% Iron Concentrate':            (62.05, 68.01),
}

DEFAULTS = {
    '% Iron Feed': 56.3, '% Silica Feed': 14.7, 'Starch Flow': 2869.0,
    'Amina Flow': 488.1, 'Ore Pulp Flow': 397.6, 'Ore Pulp pH': 9.77,
    'Flotation Column 01 Air Flow': 280.2, 'Flotation Column 02 Air Flow': 277.2,
    'Flotation Column 03 Air Flow': 281.1, 'Flotation Column 06 Air Flow': 292.1,
    'Flotation Column 07 Air Flow': 290.8, 'Flotation Column 01 Level': 520.2,
    'Flotation Column 02 Level': 522.6,  'Flotation Column 03 Level': 531.4,
    'Flotation Column 04 Level': 420.3,  'Flotation Column 05 Level': 425.3,
    'Flotation Column 06 Level': 429.9,  'Flotation Column 07 Level': 421.0,
    '% Iron Concentrate': 65.05,
}

ENG_DEFAULTS = {
    'Silica_lag_1': 2.33, 'Silica_lag_2': 2.33,
    'Iron_Concentrate_lag1': 65.05, 'Iron_Concentrate_lag2': 65.05,
    'Amina Flow_lag1': 488.1, 'Starch Flow_lag1': 2869.0,
    'Flotation Column 01 Air Flow_lag1': 280.2,
    'Flotation Column 01 Air Flow_roll_mean3': 280.2,
    'Flotation Column 01 Air Flow_roll_std3': 5.0,
    'Flotation Column 03 Air Flow_roll_mean3': 281.1,
    'Flotation Column 03 Air Flow_roll_std3': 5.0,
    'Amina_x_Col01Air': 488.1 * 280.2,
}

FEED_COLS    = ['% Iron Feed', '% Silica Feed', 'Ore Pulp Flow', 'Ore Pulp pH']
REAGENT_COLS = ['Starch Flow', 'Amina Flow']
OTHER_COLS   = ['% Iron Concentrate']