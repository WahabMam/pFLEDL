# # PFLlib: Personalized Federated Learning Algorithm Library
# # Copyright (C) 2021  Jianqing Zhang

# # This program is free software; you can redistribute it and/or modify
# # it under the terms of the GNU General Public License as published by
# # the Free Software Foundation; either version 2 of the License, or
# # (at your option) any later version.

# # This program is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# # GNU General Public License for more details.

# # You should have received a copy of the GNU General Public License along
# # with this program; if not, write to the Free Software Foundation, Inc.,
# # 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

# import h5py
# import numpy as np
# import os
# import pandas as pd


# def average_data(algorithm="", dataset="", goal="", times=10):
#     test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)

#     max_accurancy = []
#     max_accurancy_round = []
#     last_accurancy = []
#     for i in range(times):
#         max_accurancy.append(test_acc[i].max())
#         max_accurancy_round.append(test_acc[i].argmax())
#         last_accurancy.append(test_acc[i][-1])

#     print("std for best accurancy:", np.std(max_accurancy))
#     print("mean for best accurancy:", np.mean(max_accurancy))

#     print("std for best accurancy round:", np.std(max_accurancy_round))
#     print("mean for best accurancy round:", np.mean(max_accurancy_round))

#     print("std for last accurancy:", np.std(last_accurancy))
#     print("mean for last accurancy:", np.mean(last_accurancy))

#     print(f'Best acc | {round(np.mean(max_accurancy)*100, 3)}±{round(np.std(max_accurancy)*100, 3)}')
#     print(f'Best rou | {round(np.mean(max_accurancy_round), 3)}±{round(np.std(max_accurancy_round), 3)}')
#     print(f'Last rou | {round(np.mean(last_accurancy)*100, 3)}±{round(np.std(last_accurancy)*100, 3)}')

#     max_file_path = "results/" + dataset + "_" + algorithm + "_" + "max.txt"
#     # Save the maximum value to the text file
#     with open(max_file_path, 'w') as max_file:
#         max_file.write(f'best acc | {np.mean(max_accurancy)} | {np.std(max_accurancy)}')
#         max_file.write(f'\n')
#         max_file.write(f'best rou | {np.mean(max_accurancy_round)} | {np.std(max_accurancy_round)}')
#         max_file.write(f'\n')
#         max_file.write(f'last acc | {np.mean(last_accurancy)} | {np.std(last_accurancy)}')


# def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10):
#     test_acc = []
#     algorithms_list = [algorithm] * times
#     for i in range(times):
#         file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
#         test_acc.append(np.array(read_data_then_delete(file_name, delete=True)))

#     return test_acc


# def read_data_then_delete(file_name, delete=False):
#     file_path = "results/" + file_name + ".h5"
#     csv_file_path = "results/" + file_name + ".csv"

#     with h5py.File(file_path, 'r') as hf:
#         rs_test_acc = np.array(hf.get('rs_test_acc'))
#         rs_train_loss = np.array(hf.get('rs_train_loss'))
#         # rs_test_ece = np.array(hf.get('rs_test_ece')) if 'rs_test_ece' in hf  # Wahab
#         rs_test_ece = np.array(hf.get('rs_test_ece')) if 'rs_test_ece' in hf else None  # Wahab




#     # Save to CSV
#     data_dict = {
#         "Test Accuracy": rs_test_acc,
#         "Train Loss": rs_train_loss,
#         "ECE": rs_test_ece,
#     }

#     if rs_test_ece is not None:           #Wahab
#         data_dict["ECE"] = rs_test_ece    #Wahab



#     df = pd.DataFrame(data_dict)
#     df.to_csv(csv_file_path, index=False)
#     print(f"Data saved to {csv_file_path}")

#     if delete:
#         os.remove(file_path)
#     print("Length: ", len(rs_test_acc))

#     return rs_test_acc

# if __name__ == '__main__':
#     average_data(algorithm="pFedMe", dataset="Cifar100-partition_dir-clients_20-niid_True-balance_False-batch_10-classpc_10-alpha_01-train_075", goal="test", times=10)




import h5py
import numpy as np
import os
import pandas as pd

def average_data(algorithm="", dataset="", goal="", times=10):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)

    max_accuracy = []
    max_accuracy_round = []
    last_accuracy = []
    for i in range(times):
        max_accuracy.append(test_acc[i].max())
        max_accuracy_round.append(test_acc[i].argmax())
        last_accuracy.append(test_acc[i][-1])

    print("std for best accuracy:", np.std(max_accuracy))
    print("mean for best accuracy:", np.mean(max_accuracy))

    print("std for best accuracy round:", np.std(max_accuracy_round))
    print("mean for best accuracy round:", np.mean(max_accuracy_round))

    print("std for last accuracy:", np.std(last_accuracy))
    print("mean for last accuracy:", np.mean(last_accuracy))

    print(f'Best acc | {round(np.mean(max_accuracy)*100, 3)}±{round(np.std(max_accuracy)*100, 3)}')
    print(f'Best rou | {round(np.mean(max_accuracy_round), 3)}±{round(np.std(max_accuracy_round), 3)}')
    print(f'Last rou | {round(np.mean(last_accuracy)*100, 3)}±{round(np.std(last_accuracy)*100, 3)}')

    max_file_path = f"results/{dataset}_{algorithm}_max.txt"
    # Save the maximum value to the text file
    with open(max_file_path, 'w') as max_file:
        max_file.write(f'best acc | {np.mean(max_accuracy)} | {np.std(max_accuracy)}\n')
        max_file.write(f'best rou | {np.mean(max_accuracy_round)} | {np.std(max_accuracy_round)}\n')
        max_file.write(f'last acc | {np.mean(last_accuracy)} | {np.std(last_accuracy)}\n')


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = f"{dataset}_{algorithms_list[i]}_{goal}_{i}"
        test_acc.append(np.array(read_data_then_delete(file_name, delete=True)))

    return test_acc


def read_data_then_delete(file_name, delete=False):
    file_path = f"results/{file_name}.h5"
    csv_file_path = f"results/{file_name}.csv"

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))
        rs_train_loss = np.array(hf.get('rs_train_loss'))
        rs_test_ece = np.array(hf.get('rs_test_ece')) if 'rs_test_ece' in hf else None
        cen_test_acc = np.array(hf.get('cen_test_acc')) if 'cen_test_acc' in hf else None
        cen_test_auc = np.array(hf.get('cen_test_auc')) if 'cen_test_auc' in hf else None
        cen_test_loss = np.array(hf.get('cen_test_loss')) if 'cen_test_loss' in hf else None
        cen_test_ece = np.array(hf.get('cen_test_ece')) if 'cen_test_ece' in hf else None
        rs_y_prob = np.array(hf.get('y_prob')) if 'y_prob' in hf else None
        rs_y_true= np.array(hf.get('y_true')) if 'y_true' in hf else None
        round_times = np.array(hf['round_time_sec']) if 'round_time_sec' in hf else None
        cumulative_times = np.array(hf['cumulative_time_sec']) if 'cumulative_time_sec' in hf else None
        eval_gap = int(hf.attrs.get('eval_gap', 1))
        

    data_dict = {
        "acc_per": rs_test_acc,
        "loss_per": rs_train_loss,
        "ece_per": rs_test_ece,
        "acc_cen": cen_test_acc,
        "loss_cen": cen_test_loss,
        "ece_cen": cen_test_ece
    }

    # Metrics are sampled every eval_gap rounds; timing is recorded every round.
    # Series preserve missing metrics without shifting them onto the wrong rounds.
    if round_times is not None:
        data_dict = {
            key: pd.Series(values, index=np.arange(len(values)) * eval_gap, dtype=float)
            for key, values in data_dict.items() if values is not None
        }
        data_dict['round_time_sec'] = pd.Series(round_times)
        data_dict['cumulative_time_sec'] = pd.Series(cumulative_times)
        df = pd.DataFrame(data_dict).rename_axis('round').reset_index()
    else:
        # Older result files have no timing information.
        df = pd.DataFrame({
            key: pd.Series(values, dtype=float)
            for key, values in data_dict.items() if values is not None
        })
    df.to_csv(csv_file_path, index=False)

    # if delete:
    #     os.remove(file_path)

    return rs_test_acc


if __name__ == '__main__':
    average_data(
        algorithm="pFedMe",
        dataset="Cifar100-partition_dir-clients_20-niid_True-balance_False-batch_10-classpc_10-alpha_01-train_075",
        goal="test",
        times=10
    )
