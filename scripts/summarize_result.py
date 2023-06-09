"""summary of the result of the experiment.
Read the result json file and save the result in a json file.
"""

import os
import json
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import glob
from scipy import interpolate
import pdb

AP_HR = 0.62
QUERY_RATIO = [0, 50, 100, 150, 200, 300, 400, 600, 800, 1000]
# QUERY_RATIO = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 300, 400, 600, 800, 1000]
METRIC = ["AP", "AP .5", "AP .6", "AP .75", "AP .9", "AR", "AR .5", "AR .75", "AR (M)", "AR (L)"]

def load_result_json(result_dir, strategy_list, video_id_list, result_dict):
    """load result json file.
    Args:
        root_dir: root directory of the result.
        strategy_list: list of the strategies.
        video_id_list: list of the video ids.

    Returns:
        result_dict: dictionary of the result.
    """
    # load result json file for each strategy and video id
    if result_dict is None:
      result_dict = {}
    # interpolate the result
    percent1000 = np.linspace(0, 100, 1001) # 0, 0.1, 0.2, ..., 99.9, 100
    empty_dict = {}
    empty_cnt = {}
    empty_union = []
    for strategy in strategy_list:
        empty_dict[strategy] = []
        empty_cnt[strategy] = 0
        result_dict[strategy] = {"Percentage": {}, "Performance": {}, "ALC": {}, "mean_uncertainty": {}, "combine_weight": {}, "Performance_ann": {}, "ALC_ann": {}}
        print("Empty ids of", strategy, ":")
        for video_id in video_id_list:
            result_files = glob.glob(f"{result_dir}/{strategy}/{video_id}/*/result.json")
            # print(len(result_files))
            if len(result_files) == 0:
              if video_id not in empty_dict[strategy]:
                print(f"{video_id} is empty for {strategy}")
                empty_dict[strategy].append(video_id)
                empty_cnt[strategy] += 1
                if video_id not in empty_union:
                  empty_union.append(video_id)
              continue
            # print(result_files[-1])
            if os.path.exists(result_files[-1]):
              with open(result_files[-1], "r") as f:
                  result = json.load(f)
              if result["area_under_learning_curve"] in [0, -1]: # for invalid result
                continue
              # print(video_id, result["percentages"], result["performances"])
              performance1000 = interpolate.interp1d(result["percentages"], result["performances"])(percent1000)
              performance1000_ann = interpolate.interp1d(result["percentages"], result["performances_ann"])(percent1000)
              if strategy == "HP":
                result["mean_uncertaity"] = np.array(result["mean_uncertaity"])+15
              normed_unc = np.array(result["mean_uncertaity"])/result["mean_uncertaity"][0]
              result_dict[strategy]["Percentage"][video_id] = percent1000
              result_dict[strategy]["Performance"][video_id] = performance1000
              result_dict[strategy]["Performance_ann"][video_id] = performance1000_ann
              result_dict[strategy]["ALC"][video_id] = result["area_under_learning_curve"]
              result_dict[strategy]["ALC_ann"][video_id] = result["area_under_learning_curve_ann"]
              result_dict[strategy]["mean_uncertainty"][video_id] = normed_unc
              # result_dict[strategy]["combine_weight"][video_id] = result["combine_weight"] if "combine_weight" in result else np.ones(len(result["percentages"]))
              # if performance1000_ann[-1] < 100.0:
              #   print(strategy, video_id, performance1000_ann[-1])
        print("")

        # calculate the mean performance of each strategy
        result_dict[strategy]["mean_Percentage"] = percent1000
        result_dict[strategy]["Performance_all"] = np.array(result_dict[strategy]["Performance"])
        result_dict[strategy]["mean_Performance"] = np.mean(np.array(list(result_dict[strategy]["Performance"].values())), axis=0)
        result_dict[strategy]["std_Performance"] = np.std(np.array(list(result_dict[strategy]["Performance"].values())), axis=0)
        result_dict[strategy]["mean_Performance_ann"] = np.mean(np.array(list(result_dict[strategy]["Performance_ann"].values())), axis=0)
        result_dict[strategy]["Performance_all_ann"] = np.array(result_dict[strategy]["Performance_ann"])
        result_dict[strategy]["std_Performance_ann"] = np.std(np.array(list(result_dict[strategy]["Performance_ann"].values())), axis=0)
        result_dict[strategy]["mean_ALC"] = np.mean(np.array(list(result_dict[strategy]["ALC"].values())))
        result_dict[strategy]["mean_ALC_ann"] = np.mean(np.array(list(result_dict[strategy]["ALC_ann"].values())))
        result_dict[strategy]["mean_mean_uncertainty"] = np.mean(np.array(list(result_dict[strategy]["mean_uncertainty"].values())), axis=0)
        # result_dict[strategy]["mean_combine_weight"] = np.mean(np.array(list(result_dict[strategy]["combine_weight"].values())), axis=0)
        # print(f"{strategy}: {result_dict[strategy]['mean_ALC']}")
        # print(f"{strategy}: {result_dict[strategy]['mean_Percentage'][::50]}")
        # print(f"{strategy}: {result_dict[strategy]['mean_Performance'][::50]}")
    for strategy in strategy_list:
      print(f"Empty ids of {strategy}: {empty_cnt[strategy]}/{len(video_id_list)}")
    empty_dict["union"] = empty_union
    print(f"In total:\nThere are {sum(empty_cnt.values())}/{len(video_id_list)*len(strategy_list)} empty ids!")
    print(f"There are {len(empty_union)}/{len(video_id_list)} empty ids in union!")
    return result_dict, empty_dict

def summarize_result(save_dir, result_dict, ANN=False):
  # summarize the result. plot the mean performance of each strategy, and save the result in a json file.
  # plot the mean performance of each strategy. at the same time, plot it in the same figure.
  fig1, ax = plt.subplots()
  ax.set_xlabel("Labeled Samples (%)", fontsize=14)
  ax.set_ylabel("Average Precision (%)", fontsize=14)
  ax.set_xlim(0, 100)
  ax.set_ylim(60, 100)
  # graph for mean uncertainty
  fig2, ax2 = plt.subplots()
  ax2.set_ylabel("Average Uncertainty (%)", fontsize=14)
  ax2.set_xlabel("Labeled Samples (%)", fontsize=14)
  ax2.set_xlim(0, 100)

  print(result_dict.keys())
  if ANN:
    prefix = "mean_Performance_ann"
    prefix_std = "std_Performance_ann"
    prefix_alc = "mean_ALC_ann"
    prefix_pfm = "Performance_all_ann"
    save_path = "all_ANN"
  else:
    prefix = "mean_Performance"
    prefix_std = "std_Performance"
    prefix_alc = "mean_ALC"
    prefix_pfm = "Performance_all"
    save_path = "all"
  for i, strategy in enumerate(result_dict.keys()):
    initial_performance = result_dict[strategy][prefix][0]
    final_performance = result_dict[strategy][prefix][-1]
    if "THC_L1" in strategy:
        if strategy=="THC_L1_weightedfilter":
            label = "THC_L1+DWC (Ours)"
        else:
            label = strategy + "(Ours)"
        label = label.replace("THC_L1", "THC")
        linewidth = 2
        line_style = "-"
    elif "WPU_hybrid" in strategy:
        if strategy=="WPU_hybrid_weightedfilter":
            label = "WPU_hybrid+DWC (Ours)"
        else:
            label = strategy + "(Ours)"
        label = label.replace("WPU_hybrid", "WPU")
        linewidth = 2
        line_style = "-"
    elif strategy=="THC+WPU_weightedfilter":
        label = "THC+WPU+DWC (Ours)"
        linewidth = 2
        line_style = "-"
    elif strategy=="Random":
        label = strategy
        linewidth = 2
        line_style = "--"
    elif strategy=="MPE+Influence":
        label = "DC"
        linewidth = 2
        line_style = "--"
    else:
        label = strategy
        linewidth = 2
        line_style = "--"

    # plot the mean performance of each strategy
    x = result_dict[strategy]["mean_Percentage"][QUERY_RATIO]
    y= result_dict[strategy][prefix][QUERY_RATIO]
    plt.figure()
    plt.ylim(0, 100)
    # plt.fill_between(x, y - result_dict[strategy][prefix_std][QUERY_RATIO], y + result_dict[strategy][prefix_std][QUERY_RATIO], alpha=0.4)
    plt.plot(x, y, label=label, linewidth=linewidth, marker="o")
    plt.xlabel("Labeled Samples (%)")
    plt.ylabel("Average Precision (%)")
    plt.xticks(np.arange(0, 101, 5))
    plt.grid()
    plt.legend()
    if not os.path.exists(os.path.join(save_dir, strategy)):
        os.mkdir(os.path.join(save_dir, strategy))
    plt.savefig(os.path.join(save_dir, strategy, f"{strategy}_performance.png"))
    plt.close()

    # ax.fill_between(x, y - result_dict[strategy][prefix_std][QUERY_RATIO], y + result_dict[strategy][prefix_std][QUERY_RATIO], alpha=0.2)
    c = plt.get_cmap("tab10")(i)
    ax.plot(x, y, label=label, linewidth=linewidth, linestyle=line_style, color=c, marker="o", markersize=5)

    # plot the mean uncertainty of each strategy
    y = np.array(result_dict[strategy]["mean_mean_uncertainty"]) * 100
    c = plt.get_cmap("tab10")(i+1)
    ax2.plot(x, y, label=label, linewidth=linewidth, linestyle=line_style, color=c, marker="o", markersize=5)
    plt.figure()
    plt.plot(x, y, label=label, linewidth=linewidth, marker="o")
    plt.xlabel("Labeled Samples (%)")
    plt.ylabel("Average Uncertainty (%)")
    plt.xticks(np.arange(0, 101, 5))
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(save_dir, strategy, f"{strategy}_uncertainty.png"))
    plt.close()

  # plot the mean performance of each strategy in the same figure
  ax.set_xticks(np.array(QUERY_RATIO)/10)
  ax.set_yticks(np.arange(60, 101, 5))
  ax.tick_params(labelsize=12)
  ax.legend(fontsize=14)
  ax.grid()
  fig1.savefig(os.path.join(save_dir, f"{save_path}_performance.png"))
  fig1.savefig(os.path.join(save_dir, f"{save_path}_performance.pdf"))

  # save ax2
  ax2.set_xticks(np.array(QUERY_RATIO)/10)
  ax2.set_yticks(np.arange(70, 125, 5))
  ax2.tick_params(labelsize=12)
  ax2.legend(fontsize=14)
  ax2.grid()
  fig2.savefig(os.path.join(save_dir, f"{save_path}_uncertainty.png"))
  fig2.savefig(os.path.join(save_dir, f"{save_path}_uncertainty.pdf"))

  metric_dict = {}
  for strategy in result_dict.keys():
      metric_dict[strategy] = {}
      # convert each np.arrays to list, so that it can be saved in a json file
      # converted_percentage = result_dict[strategy]["mean_Percentage"].tolist()[::50]
      converted_performance = np.array(result_dict[strategy][prefix].tolist())
      # pdb.set_trace()
      converted_alc = float(result_dict[strategy][prefix_alc])
      metric_dict[strategy]["mean_Percentage"] = QUERY_RATIO
      metric_dict[strategy][prefix] = converted_performance[QUERY_RATIO].tolist()
      metric_dict[strategy][prefix_alc] = converted_alc
      print(f"{strategy} ALC: {converted_alc}")
      metric_dict[strategy]["mean_mean_uncertainty"] = result_dict[strategy]["mean_mean_uncertainty"].tolist()
      # metric_dict[strategy]["mean_combine_weight"] = result_dict[strategy]["mean_combine_weight"].tolist()
  # save the result in a json file
  if ANN:
    with open(os.path.join(save_dir, "metric_ann.json"), "w") as f:
      json.dump(metric_dict, f, indent=4)
  else:
    with open(os.path.join(save_dir, "metric.json"), "w") as f:
        json.dump(metric_dict, f, indent=4)

def main(name):
  root_dir = "exp"
  if "MVA" in name:
    video_id_list_path = "configs/val_video_list_full.txt"
    strategy_list = {name: ["Random", "HP", "MPE+Influence", "TPC", "THC_L1_weightedfilter", "WPU_hybrid_weightedfilter", "THC+WPU_weightedfilter"]}
    # strategy_list = {name: ["HP", "MPE+Influence", "TPC", "THC_L1_weightedfilter", "WPU_hybrid_weightedfilter"]}
  elif "PCIT" in name:
    video_id_list_path = "configs/PCIT_video_list.txt"
    strategy_list = {name: ["Random", "HP", "TPC", "MPE+Influence", "THC_L1_weightedfilter", "WPU_hybrid_weightedfilter"]}

  # video_id_list_path = "configs/trainval_video_list.txt"

  # read video id list
  with open(video_id_list_path, "r") as f:
      video_id_list = f.read().splitlines()
  # load result json file
  print("loading result json file...")
  for exp_name in strategy_list.keys():
    result_dict = None
    print(f"loading {exp_name}...")
    result_dir = os.path.join(root_dir, f"AL_{exp_name}", "SimplePose")
    result_dict, empty_dict = load_result_json(result_dir, strategy_list[exp_name], video_id_list, result_dict)

    save_dir = os.path.join(root_dir, "results", f"{exp_name}") #exp
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    with open(os.path.join(save_dir, "empty_dict.json"), "w") as f:
        json.dump(empty_dict, f, indent=4)
    print("done\n")

    # summarize the result
    print("summarizing the result w/o annotation...")
    if not os.path.exists(os.path.join(save_dir, "RAW")):
        os.makedirs(os.path.join(save_dir, "RAW"))
    summarize_result(os.path.join(save_dir, "RAW"), result_dict, ANN=False)

    print("\nsummarizing the result with annotation...")
    if not os.path.exists(os.path.join(save_dir, "ANN")):
      os.makedirs(os.path.join(save_dir, "ANN"))
    summarize_result(os.path.join(save_dir, "ANN"), result_dict, ANN=True)
    print("Done!\n")

if __name__ == "__main__":
    main(name="PCIT2")