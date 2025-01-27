import torch
from run import RunSimulation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Le code se lance sur {device}")


folder_result_name = "1_piche"  # name of the result folder


# On utilise hyper_param_init uniquement si c'est un nouveau modèle


hyper_param_init = {
    "H": [
        230.67,
        230.67,
        # 230.67,
        # 230.67,
        # 230.67,
        # 261.39,
        # 261.39,
        # 261.39,
        # 261.39,
        # 261.39,
    ],  # la rigidité du ressort
    "ya0": [
        0.00125,
        0.00375,
        # 0.00625,
        # 0.00875,
        # 0.00125,
        # 0.00375,
        # 0.00625,
        # 0.00875,
    ],  # la position initiale du ressort
    "m": 1.57,  # la masse du ressort
    "file": [
        "data_john_4_case_2.csv",
        "data_john_5_case_2.csv",
        # "data_john_7_case_2.csv",
        # "data_john_9_case_2.csv",
        # "data_john_4_case_1.csv",
        # "data_john_5_case_1.csv",
        # "data_john_7_case_1.csv",
        # "data_john_9_case_1.csv",
    ],
    "nb_epoch": 1000,  # epoch number
    "save_rate": 20,  # rate to save
    "dynamic_weights": True,
    "lr_weights": 1e-1,  # si dynamic weights
    "weight_data": 0.33,
    "weight_border": 0.33,
    "batch_size": 10000,  # a changer
    "Re": 100,
    "lr_init": 0.001,
    "gamma_scheduler": 0.999,  # pour la lr
    "nb_layers": 15,
    "nb_neurons": 64,
    "n_data_test": 5000,
    "nb_points": 12 * 12,  # le nombre de points pris par axe par pas de temps
    "x_min": -0.1,
    "x_max": 0.1,
    "y_min": -0.06,
    "y_max": 0.06,
    "t_min": 6.5,
    "nb_period": 10,
    "nb_period_plot": 2,
    "nb_points_close_cylinder": 50,  # le nombre de points proches du cylindre
    "rayon_close_cylinder": 0.015,
    "nb_points_border": 25,  # le nombre de points sur la condition init
    "force_inertie_bool": True,
}


param_adim = {"V": 1.0, "L": 0.025, "rho": 1.2}

simu = RunSimulation(hyper_param_init, folder_result_name, param_adim)

simu.run()
