"""summary of the result of the experiment.
Read the result json file and save the result in a json file.
"""

import os
import json
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
import seaborn as sns
import numpy as np
import glob
from scipy import interpolate
import pdb
from active_learning.al_metric import plot_learning_curves, compute_alc
import japanize_matplotlib

AP_HR = 0.62
QUERY_RATIO = [0, 50, 100, 150, 200, 300, 400, 600, 800, 1000]
# QUERY_RATIO = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# METRIC = ["AP", "AP .5", "AP .6", "AP .7", "AP .75", "AP .8"]
# METRIC = ["AP"]
METRIC = ["AP .6", "AP .75"]

def load_result_json(result_dir, strategy_list, video_id_list, result_dict, sc_thresh):
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
    num_valid = 0
    for strategy in strategy_list:
        empty_dict[strategy] = []
        empty_cnt[strategy] = 0
        result_dict[strategy] = {"Percentage": {}, "mean_uncertainty": {}, "combine_weight": {}, "spearmanr": {}, "actual_finish": {}, "finished_minerror": {}, "finished_oursc": {}, "stopped_AP_min": {}, "stopped_AP_oursc": {}}
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
              # print(strategy, video_id)
              result_dict[strategy]["Percentage"][video_id] = percent1000
              performance_dict = result["performances"]
              performance_dict_ann = result["performances_ann"]
              for m, metric in enumerate(METRIC):
                if not str(metric) in result_dict[strategy].keys():
                  result_dict[strategy][str(metric)] = {}
                  result_dict[strategy][str(metric)+"_ann"] = {}
                  result_dict[strategy][str(metric)+"_ALC"] = {}
                  result_dict[strategy][str(metric)+"_ALC_ann"] = {}
                  result_dict[strategy][str(metric)+"_mean"] = {}
                  result_dict[strategy][str(metric)+"_std"] = {}
                  result_dict[strategy][str(metric)+"_mean_ann"] = {}
                  result_dict[strategy][str(metric)+"_std_ann"] = {}
                  result_dict[strategy][str(metric)+"_ALC_mean"] = {}
                  result_dict[strategy][str(metric)+"_ALC_mean_ann"] = {}
                performance = np.array(list(round_res[metric] for round_res in performance_dict))*100
                performance_ann = np.array(list(round_res[metric] for round_res in performance_dict_ann))*100
                if -1 in performance or -1 in performance_ann:
                    continue
                # elif performance_ann[-1] != 100 and m==0:
                    # print(f"{strategy} {video_id} is not 100%! final AP: performance_ann[-1])
                num_valid += 1
                performance1000 = interpolate.interp1d(result["percentages"], performance)
                performance1000_ann = interpolate.interp1d(result["percentages"], performance_ann)
                result_dict[strategy][str(metric)][video_id] = performance1000(percent1000)
                result_dict[strategy][str(metric)+"_ann"][video_id] = performance1000_ann(percent1000)
                result_dict[strategy][str(metric)+"_ALC"][video_id] = compute_alc(result["percentages"], performance)
                result_dict[strategy][str(metric)+"_ALC_ann"][video_id] = compute_alc(result["percentages"], performance_ann)
              if strategy == "HP":
                result["mean_uncertaity"] = np.array(result["mean_uncertaity"])+15
              elif result["mean_uncertaity"][0] == 0:
                result["mean_uncertaity"] = np.array(result["mean_uncertaity"])+1
              normed_unc = np.array(result["mean_uncertaity"])/result["mean_uncertaity"][0]
              result_dict[strategy]["mean_uncertainty"][video_id] = normed_unc
              if "spearmanr" in result.keys():
                result_dict[strategy]["spearmanr"][video_id] = result["spearmanr"]
              if sc_thresh != None:
                result_dict[strategy]["actual_finish"][video_id] = result["actual_finish"]
                result_dict[strategy]["finished_minerror"][video_id] = result["finished_minerror"]
                result_dict[strategy]["finished_oursc"][video_id] = result["finished_oursc"]
                stopped_min = find_nearest(result["percentages"], result["finished_minerror"])
                stopped_oursc = find_nearest(result["percentages"], result["finished_oursc"])
                result_dict[strategy]["stopped_AP_min"][video_id] = performance_dict_ann[stopped_min][sc_thresh]
                result_dict[strategy]["stopped_AP_oursc"][video_id] = performance_dict_ann[stopped_oursc][sc_thresh]
        # sort the ALC of each video id
        # print(result_dict[strategy].keys())
        # ALC_dict = result_dict[strategy][str("AP .75") + "_ALC_ann"]
        # sorted_ALC = sorted(ALC_dict.items(), key=lambda x:x[1])
        # print(sorted_ALC[:10])

        # evaluate stopping criterion
        if sc_thresh != None:
            mean_actual_finish = np.mean(np.array(list(result_dict[strategy]["actual_finish"].values())))
            mean_finished_minerror = np.mean(np.array(list(result_dict[strategy]["finished_minerror"].values())))
            mean_finished_oursc = np.mean(np.array(list(result_dict[strategy]["finished_oursc"].values())))
            mean_stopped_AP_min = np.mean(np.array(list(result_dict[strategy]["stopped_AP_min"].values())))
            mean_stopped_AP_oursc = np.mean(np.array(list(result_dict[strategy]["stopped_AP_oursc"].values())))
            print(f"{strategy} actual_finish: {mean_actual_finish}")
            print(f"{strategy} finished_minerror: {mean_finished_minerror}")
            print(f"{strategy} finished_oursc: {mean_finished_oursc}")
            print(f"{strategy} stopped_AP_min: {mean_stopped_AP_min}")
            print(f"{strategy} stopped_AP_oursc: {mean_stopped_AP_oursc}")

        # calculate the mean performance of each strategy
        result_dict[strategy]["mean_Percentage"] = percent1000
        for metric in METRIC:
            # result_dict[strategy][str(metric)+"_all"] = np.array(result_dict[strategy][str(metric)])
            result_dict[strategy][str(metric)+"_mean"] = np.mean(np.array(list(result_dict[strategy][str(metric)].values())), axis=0)
            result_dict[strategy][str(metric)+"_std"] = np.std(np.array(list(result_dict[strategy][str(metric)].values())), axis=0)
            result_dict[strategy][str(metric)+"_mean_ann"] = np.mean(np.array(list(result_dict[strategy][str(metric)+"_ann"].values())), axis=0)
            # result_dict[strategy][str(metric)+"_all_ann"] = np.array(result_dict[strategy][str(metric)+"_ann"])
            result_dict[strategy][str(metric)+"_std_ann"] = np.std(np.array(list(result_dict[strategy][str(metric)+"_ann"].values())), axis=0)
            result_dict[strategy][str(metric)+"_ALC_mean"] = np.mean(np.array(list(result_dict[strategy][str(metric)+"_ALC"].values())))
            result_dict[strategy][str(metric)+"_ALC_mean_ann"] = np.mean(np.array(list(result_dict[strategy][str(metric)+"_ALC_ann"].values())))

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
    print(f"--> There are {int(num_valid/len(METRIC)/len(strategy_list))} valid ids in average!")
    return result_dict, empty_dict

def summarize_result(save_dir, result_dict, metric, ANN=False):
  # summarize the result. plot the mean performance of each strategy, and save the result in a json file.
  # plot the mean performance of each strategy. at the same time, plot it in the same figure.
  fig1, ax = plt.subplots(nrows=2, sharex='col', gridspec_kw={'height_ratios': (11,1)} )
  # ax[0].set_xlabel("Labeled Samples (%)", fontsize=20)
  metlabel = metric.replace(" .", "@ .")
  ax[1].set_xlabel("Labeled Percentage (%)", fontsize=20)
  ax[0].set_ylabel(metlabel+" (%)", fontsize=20)
  fig1.subplots_adjust(hspace=0.0, left=0.15, bottom=0.15)

  fig2, ax2 = plt.subplots() # graph for mean uncertainty
  ax2.set_ylabel("Average Uncertainty (%)", fontsize=14)
  ax2.set_xlabel("AP@.6 (%)", fontsize=14)
  ax2.set_xlim(75, 100)
  print(f"{metric}", list(result_dict.keys()))

  prefix = "_ann" if ANN else ""
  ymin = 100 # initial value
  for i, strategy in enumerate(result_dict.keys()):
    if "THC" in strategy or "WPU" in strategy:
        line_style = "-"
        linewidth = 2
        label = (strategy + " (Ours)").replace("_", "+").replace("filter", "").replace("weighted", "DWC").replace("Coreset", "DUW")
        c = plt.get_cmap("Set1")(0)
        if strategy=="THC":
          c = plt.get_cmap("Set1")
        elif strategy=="WPU":
          c = plt.get_cmap("Set1")
    else:
        line_style = "--"
        linewidth = 2
        label = strategy
        if strategy=="Random":
            label = "Random"
            c = "black"
        elif strategy=="HP":
            label = "LC"
            c = plt.get_cmap("Set1")(2)
        elif strategy=="MPE+Influence":
            label = "DC"
        elif strategy=="MPE":
            c = "chocolate"
        elif strategy=="TPC":
            c = plt.get_cmap("Set1")(3)
        elif "filter" in strategy:
            label = strategy.replace("filter", "").replace("_", "")
            label = label.replace("K-Means", "k-means")
            label = label.replace("Coreset", "Core-Set")
            if strategy=="_K-Meansfilter":
              c = plt.get_cmap("Set1")(1)
            elif strategy=="_Coresetfilter":
              c = "gold"

    # plot the mean performance of each strategy
    x = result_dict[strategy]["mean_Percentage"][QUERY_RATIO]
    y= np.array(result_dict[strategy][str(metric)+"_mean"+prefix])[QUERY_RATIO]
    plt.figure()
    plt.ylim(0, 100)
    if ymin > y[0]:
        ymin = y[0]
    # plt.fill_between(x, y - result_dict[strategy][prefix_std][QUERY_RATIO], y + result_dict[strategy][prefix_std][QUERY_RATIO], alpha=0.4)
    plt.plot(x, y, label=label, linewidth=linewidth, marker="o")
    plt.xlabel("Labeled Samples (%)")
    plt.ylabel(str(metric)+" (%)")
    plt.xticks(np.arange(0, 101, 5))
    plt.grid()
    plt.legend()
    if not os.path.exists(os.path.join(save_dir, strategy)):
        os.mkdir(os.path.join(save_dir, strategy))
    plt.savefig(os.path.join(save_dir, strategy, f"{strategy}_{str(metric)+prefix}.png"))
    plt.close()

    # ax.fill_between(x, y - result_dict[strategy][str(metric)+"_std"+prefix][QUERY_RATIO], y + result_dict[strategy][str(metric)+"_std"+prefix][QUERY_RATIO], alpha=0.3)
    ax[0].plot(x, y, label=label, linewidth=linewidth, linestyle=line_style, color=c, marker="o", markersize=5)

    x = np.array(result_dict[strategy]["AP .6"+"_mean"+prefix])[QUERY_RATIO]
    # plot the mean uncertainty of each strategy
    y = np.array(result_dict[strategy]["mean_mean_uncertainty"]) * 100
    # if uncertainty is always 100, skip it
    if np.all(y==100):
        continue
    ax2.plot(x, y, label=label, linewidth=linewidth, linestyle=line_style, color=c, marker="o", markersize=5)
    # 矢印を追加
    for i in range(len(x) - 1):
      dx = x[i+1] - x[i]  # xの差分
      dy = y[i+1] - y[i]  # yの差分
      # ax2.arrow(x[i], y[i], dx / 2, dy / 2, color=c, head_width=1.2)
      arrow = FancyArrowPatch((x[i], y[i]), (x[i]+dx/2, y[i]+dy/2), arrowstyle='-|>', mutation_scale=20, color=c)
      ax2.add_patch(arrow)
    plt.figure()
    plt.plot(x, y, label=label, linewidth=linewidth, marker="o")
    plt.ylabel("Average Uncertainty (%)")
    plt.xticks(np.arange(60, 101, 2))
    plt.grid()   # plt.xlabel("Labeled Samples (%)")
    plt.legend()
    plt.savefig(os.path.join(save_dir, strategy, f"{strategy}_uncertainty.png"))
    plt.close()

  # plot the mean performance of each strategy in the same figure
  ax[0].set_ylim(5*int(ymin/5)-3, 100)
  ax[0].set_xticks(np.array(QUERY_RATIO)/10)
  ax[0].set_yticks(np.arange(5*int(ymin/5), 101, 5))
  ax[1].set_ylim(0, 10)
  ax[1].set_yticks([0])
  ax[0].tick_params(labelsize=14)
  ax[1].tick_params(labelsize=14)
  ax[0].legend(fontsize=14)
  ax[0].grid()
  ax[1].grid()

  # 省略を意味する波線の描画
  d1 = 0.01 # X軸のはみだし量
  d2 = 0.2 # ニョロ波の高さ
  wn = 21   # ニョロ波の数（奇数値を指定）
  pp = (0,d2,0,-d2)
  px = np.linspace(-d1,1+d1,wn)
  py = np.array([1+pp[i%4] for i in range(0,wn)])
  p = Path(list(zip(px,py)), [Path.MOVETO]+[Path.CURVE3]*(wn-1))
  line1 = mpatches.PathPatch(p, lw=5, edgecolor='black',
                            facecolor='None', clip_on=False,
                            transform=ax[1].transAxes, zorder=10)
  line2 = mpatches.PathPatch(p,lw=4, edgecolor='white',
                            facecolor='None', clip_on=False,
                            transform=ax[1].transAxes, zorder=10,
                            capstyle='round')
  ax[1].add_patch(line1)
  ax[1].add_patch(line2)

  fig1.savefig(os.path.join(save_dir, f"{str(metric)+prefix}.png"))
  fig1.savefig(os.path.join(save_dir, f"{str(metric)+prefix}.pdf"))
  plt.close(fig1)

  # save ax2
  ax2.set_xticks(np.arange(75, 101, 5))
  ax2.set_yticks(np.arange(70, 185, 10))
  ax2.tick_params(labelsize=12)
  ax2.legend(fontsize=12)
  ax2.grid()
  fig2.savefig(os.path.join(save_dir, f"uncertainty.png"))
  fig2.savefig(os.path.join(save_dir, f"uncertainty.pdf"))
  plt.close(fig2)

  metric_dict = {}
  for strategy in result_dict.keys():
      metric_dict[strategy] = {}
      # convert each np.arrays to list, so that it can be saved in a json file
      # converted_percentage = result_dict[strategy]["mean_Percentage"].tolist()[::50]
      converted_performance = np.array(result_dict[strategy][str(metric)+"_mean"+prefix].tolist())
      # pdb.set_trace()
      converted_alc = float(result_dict[strategy][str(metric)+"_ALC_mean"+prefix])
      metric_dict[strategy]["mean_Percentage"] = QUERY_RATIO
      metric_dict[strategy][str(metric)+prefix] = converted_performance[QUERY_RATIO].tolist()
      metric_dict[strategy][str(metric)+"_ALC"] = converted_alc
      print(f"{strategy} ALC: {converted_alc}")
      metric_dict[strategy]["mean_mean_uncertainty"] = result_dict[strategy]["mean_mean_uncertainty"].tolist()
      # metric_dict[strategy]["mean_combine_weight"] = result_dict[strategy]["mean_combine_weight"].tolist()
  print("Done!\n")
  return metric_dict

def plot_spearman(save_dir, result_dict):
  """compute the mean spearmanr of each strategy, and plot the result."""
  plt.figure()
  percentage = np.array(QUERY_RATIO)/10
  for i, strategy in enumerate(result_dict.keys()):
    x = np.array(result_dict[strategy]["AP .75"+"_mean"])[QUERY_RATIO]
    spearmanr = np.array(list(result_dict[strategy]["spearmanr"].values())).mean(axis=0)
    if len(spearmanr) == 0:
      continue
    print(f"{strategy}: {spearmanr}")
    plt.plot(percentage, spearmanr, label=strategy, linewidth=2, marker="o")
  plt.xlabel("Labeled Samples (%)")
  plt.ylabel("Spearmanr")
  plt.xticks(np.arange(60, 101, 5))
  plt.grid()
  plt.legend()
  plt.savefig(os.path.join(save_dir, "spearmanr.png"))
  plt.close()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def main(video_id_list_path, strategy_list, sc_thresh, RAW=False, ANN=True):
  root_dir = "exp"
  # video_id_list_path = "configs/trainval_video_list.txt"

  # read video id list
  with open(video_id_list_path, "r") as f:
      video_id_list = f.read().splitlines()
  # load result json file
  print("loading result json file...")
  result_dict = None
  for exp_name in strategy_list.keys():
    print(f"loading {exp_name}...")
    result_dir = os.path.join(root_dir, f"AL_{exp_name}", "SimplePose")
    result_dict, empty_dict = load_result_json(result_dir, strategy_list[exp_name], video_id_list, result_dict, sc_thresh)
    save_dir = os.path.join(root_dir, "results", f"{exp_name}") #exp
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    with open(os.path.join(save_dir, "empty_dict.json"), "w") as f:
        json.dump(empty_dict, f, indent=4)
    print("done\n")

  # summarize the result
  if RAW:
      print("summarizing the result w/o annotation...")
      if not os.path.exists(os.path.join(save_dir, "RAW")):
          os.makedirs(os.path.join(save_dir, "RAW"))
      for metric in METRIC:
          summarize_result(os.path.join(save_dir, "RAW"), result_dict, metric, ANN=False)
      with open(os.path.join(save_dir, f"{metric}.json"), "w") as f:
          json.dump(metric_dict, f, indent=4)
  if ANN:
      result_ann_dict = {}
      print("\nsummarizing the result with annotation...")
      if not os.path.exists(os.path.join(save_dir, "ANN")):
        os.makedirs(os.path.join(save_dir, "ANN"))
      for metric in METRIC:
          metric_dict = summarize_result(os.path.join(save_dir, "ANN"), result_dict, metric, ANN=True)
          result_ann_dict[metric] = metric_dict
      # save the result in a json file
      with open(os.path.join(save_dir, f"result_ann.json"), "w") as f:
        json.dump(result_ann_dict, f, indent=4)
  # plot_spearman(save_dir, result_dict)
  print("Done!\n")

if __name__ == "__main__":
    is_PoseTrack = True
    is_JRDB = False

    if is_PoseTrack:
      video_id_list_path = "configs/val_video_list_full.txt"
    elif is_JRDB:
      video_id_list_path = "configs/jrdb-pose/val_ids.txt"
    else:
      video_id_list_path = "configs/PCIT_video_list.txt"

    strategy_list = {"WACV_ACFT": ["Random", "HP", "MPE", "TPC", "_K-Meansfilter", "_Coresetfilter"], "WACV_DUW0.01": ["THC+WPU_Coresetfilter"]}
    main(video_id_list_path, strategy_list, sc_thresh=None, RAW=False, ANN=True)