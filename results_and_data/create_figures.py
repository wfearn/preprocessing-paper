"""
This script takes the data file and creates the figures and tables used in the paper except for the vocab curve plot
It will put the tables in a `tables` folder, with full tables for each classifier, hash and rare words tables, and the high level table
The data this script uses is in the `results_100k.csv` file
"""

import os
import glob
import re
import copy
import pickle
from collections import namedtuple

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind_from_stats

plt.style.use('ggplot')

rare_mapping = {
    "1": 1,
    "8": 2,
    "50": 3,
    "345": 4,
    "2419": 5,
    "16957": 6,
    "118903": 7,
    "833786": 8,
    "5846811": 9,
    "41000000": 10,
}

metric = "accuracy" # or `f1`, both are similar

def write_to_latex(df, file_name: str, main_table: bool = True):
    df["method"] = df["method"].apply(lambda x: x.replace("h40000", "hash").replace("r118903", "rare").replace(",", "+"))
    df["method"] = df["method"].apply(lambda x: x.replace("h40000", "hash").replace("r972944", "rare"))

    df = prettify_table_latex(df, main_table)
    if not os.path.isdir("tables"):
        os.makedirs("tables")

    with open("tables/" + file_name,'w') as tf:
        tf.write(df.to_latex(index=False).replace("\\textbackslash ", "\\").replace("\\{", "{").replace("\\}", "}"))

def prettify_table_latex(df, main_table=True) -> pd.DataFrame:
    ### Start by making the std attached ###
    og_df = copy.deepcopy(df)
    for column in df:
        if "std" in column:
            df[column.replace(" std",  "")] = df[column.replace(" std",  "")].astype(str)
            df[column.replace(" std",  "")] += df[column].apply(lambda x: "  \textpm\ " + str(x))
    df_cols_no_std = [col for col in df.columns if "std" not in col]
    final_df = df[df_cols_no_std]

    ### Highlight the statistically significant values ###
    if main_table:
        ind_methods = og_df.iloc[:10]
        lowest_train_time = og_df.iloc[10:15]
        highest_acc = og_df.iloc[15:]
        zip_obj = zip([ind_methods, lowest_train_time, highest_acc], [0, 10, 15])
    else:
        zip_obj = zip([og_df], [0])

    for given_df, df_offset in zip_obj:
        for col_name, is_min in zip(["train time", "test time", "vocab size", metric], [True, True, True, False]):
            min_val_idx = given_df[col_name].argmin() if is_min else given_df[col_name].argmax()
            min_val = given_df[col_name].iloc[min_val_idx]
            for index, (_, (row)) in enumerate(given_df.iterrows()):
                stat, p_val = ttest_ind_from_stats(min_val, given_df[col_name + " std"].iloc[min_val_idx], 5, row[col_name], row[col_name + " std"], 5, equal_var=False)
                if p_val > 0.05 or pd.isnull(p_val): # get nans as zeros
                    cur_text = final_df.iloc[df_offset + index][col_name]
                    cur_text_mean = cur_text.split(" ")[0]
                    cur_text_mean = "\textbf{" + cur_text_mean + "}"
                    final_bold_val = " ".join([cur_text_mean] + cur_text.split(" ")[1:])
                    final_df.iloc[df_offset + index][col_name] = final_bold_val

    ### Make the Section Headings and capitalize ###
    if main_table:
        type_map = {
            4: "Individual Methods",
            12: "Lowest Train/Test Time",
            17: "Highest Accuracy",
        }
        type_col = ["" if i not in type_map else type_map[i] for i in range(len(final_df))]
        final_df["Type"] = type_col
        final_df = final_df[["Type"] + list(final_df.columns[:-1])]
        final_df.columns = [item.title() for item in final_df.columns]

    else:
        final_df.columns = [item.title() for item in final_df.columns]

    return final_df

def get_ave_results(df: pd.DataFrame, dataset_name: str, enforce_single: bool = False,
                        specific_model: str = None) -> pd.DataFrame:
        KEEP_COLS = [
            "vocabsize",
            "vocabsize_std",
            "traintime",
            "testtime",
            "dataset",
            "method",
            metric,
            f"{metric}_std",
            "traintime_std",
            "testtime_std"
        ]
        train_col = "traintime"
        test_col = "testtime"

        table = []
        SAVED_METHODS = []
        top_df = copy.deepcopy(df)
        baselines_idx = df.preprocessing_methods.apply(lambda x: pd.isnull(x))
        baselines = top_df[baselines_idx]
        top_df = top_df[~baselines_idx] # not baselines
        for dataset, dataset_df in top_df.groupby("corpus"):
            if dataset_name not in dataset: continue
            for method, method_df in dataset_df.groupby("preprocessing_methods"):
                if enforce_single and (len(method.split(",")) != 1 and method not in SAVED_METHODS):
                    continue

                model_results = []

                for model, model_df in method_df.groupby("model"):

                    if not specific_model:
                        if model in ["ankura"]:
                            continue
                    else:
                        if model != specific_model:
                            continue

                    new_data = copy.deepcopy(model_df)

                    baseline_vals = baselines[(baselines.model == model) & (baselines.corpus == dataset)]
                    if baseline_vals.empty:
                        print(f"No baselines vals for dataset {dataset} and model {model} and method {method}")
                        continue
                    baseline_mean = baseline_vals[metric].mean()
                    baseline_std = baseline_vals[metric].std()

                    for col in new_data.columns:
                        if col not in ["corpus", "preprocessing_methods", "doc_size", "model", "seed"]:
                            new_data["norm_" + col] = new_data[col]
                            new_data[col] = new_data[col] / baseline_vals.mean()[col].item()

                    ave_result = new_data.mean()
                    ave_result[f"{metric}_std"] =  new_data[metric].std()

                    ave_result["traintime"] =  new_data[train_col].mean()
                    ave_result["testtime"] =  new_data[test_col].mean()

                    ave_result["traintime_std"] =  new_data[train_col].std()
                    ave_result["testtime_std"] =  new_data[test_col].std()
                    ave_result["vocabsize_std"] =  new_data["vocabsize"].std()

                    ave_result["dataset"] = dataset
                    ave_result["method"] = method
                    ave_result = ave_result
                    model_results.append(ave_result)

                if len(model_results):
                    method_results = pd.concat(model_results, axis=1).transpose()
                    final_row = method_results.iloc[0]
                    for col in method_results.columns:
                        if col not in ["model", "dataset", "method"]:
                            final_row[col] = method_results[col].mean()
                    table.append(final_row)

        table = pd.DataFrame(table)
        col_order = ['method', 'vocabsize', "vocabsize_std", 'traintime', 'traintime_std', 'testtime', 'testtime_std', metric, f'{metric}_std']
        col_rename = ['method', 'vocab size', "vocab size std", 'train time', 'train time std', 'test time', 'test time std', metric, f'{metric} std']
        table = table[col_order]
        for col in table.columns:
            if col != "method":
                table[col] = (table[col] * 100).round(1)
        table.columns = col_rename
        return table.sort_values("train time")


def main():
    df = pd.read_csv("results_100k.csv", index_col=0, header=0)

    # gather non topics
    all_models = ['vowpal', 'svm', 'naive', 'knn']

    amazon_results = get_ave_results(df, "amazon")

    # Save all results for appendix:
    write_to_latex(copy.deepcopy(amazon_results), 'amazon_table_FULL.tex', main_table=False)

    # Get selected results
    one_method = amazon_results[~amazon_results.method.str.contains(",")]
    two_plus_methods = amazon_results[amazon_results.method.str.contains(",")]
    two_plus_methods = pd.concat([two_plus_methods.iloc[:5], two_plus_methods.sort_values(metric).iloc[-5:]], axis=0)
    amazon_results = pd.concat([one_method, two_plus_methods], axis=0)
    write_to_latex(amazon_results, "amazon_table.tex")

    ## Each Model:
    for model in all_models:
        amazon_results = get_ave_results(df, "amazon", specific_model=model)
        write_to_latex(copy.deepcopy(amazon_results), f'amazon_table_FULL_{model}.tex', main_table=False)


    ap_results = get_ave_results(df, "ap")
    # write full results
    write_to_latex(copy.deepcopy(ap_results), 'ap_table_FULL.tex', main_table=False)
    # save methods for paper
    one_method = ap_results[~ap_results.method.str.contains(",")]
    two_plus_methods = ap_results[ap_results.method.str.contains(",")]
    two_plus_methods = pd.concat([two_plus_methods.iloc[:5], two_plus_methods.sort_values(metric).iloc[-5:]], axis=0)
    ap_results = pd.concat([one_method, two_plus_methods], axis=0)
    write_to_latex(ap_results, 'ap_table.tex')

    for model in all_models:
        ap_results = get_ave_results(df, "ap", specific_model=model)
        write_to_latex(copy.deepcopy(ap_results), f'ap_table_FULL_{model}.tex', main_table=False)


    ### Get values for rare word hashing and rare word filtering ###
    df_all = df
    amazon_results = get_ave_results(df_all, "amazon")
    rare_scores = amazon_results[amazon_results.method.apply(lambda x: x[0] == "r" and "," not in x)]
    write_to_latex(copy.deepcopy(rare_scores), f'amazon_table_rare.tex', main_table=False)
    hash_scores = amazon_results[amazon_results.method.apply(lambda x: x[0] == "h" and "," not in x)]
    write_to_latex(copy.deepcopy(hash_scores), f'amazon_table_hash.tex', main_table=False)

    ap_results = get_ave_results(df_all, "ap")
    rare_scores = ap_results[ap_results.method.apply(lambda x: x[0] == "r" and "," not in x)]
    write_to_latex(copy.deepcopy(rare_scores), f'ap_table_rare.tex', main_table=False)
    hash_scores = ap_results[ap_results.method.apply(lambda x: x[0] == "h" and "," not in x)]
    write_to_latex(copy.deepcopy(hash_scores), f'ap_table_hash.tex', main_table=False)

    amazon_sizes = df[df.corpus == "amazon"].doc_size.unique()
    size_dict = {}
    for doc_size in amazon_sizes:
        baselines = df[(df.corpus == "amazon") & (df.doc_size == doc_size) & (df.preprocessing_methods.apply(lambda x: pd.isnull(x)))]
        if baselines.empty:
            continue

        try:
            size_dict[doc_size] = get_ave_results(df, "amazon", enforce_single=True)
        except Exception as e:
            print(f"Failed on doc size {doc_size}")
            continue

    for doc_size_o in size_dict.keys():
        curve_table = []
        corr_df = []
        for doc_size, doc_df in size_dict.items():
            if doc_size not in [doc_size_o]:
                continue
            for (index, row) in doc_df.iterrows():
                curve_table.append({
                    "num docs": int(int(doc_size) / 1000),
                    "train time": row["train time"],
                    "test time": row["test time"],
                    "vocab": row["vocab size"],
                    "metric": row[metric],
                    "method": row["method"]
                })
        curve_df = pd.DataFrame(curve_table)

        ax = sns.heatmap(curve_df.drop(["method", "num docs"], axis=1).corr().round(2), annot=True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(f"heatmap_normalized_{doc_size_o}.png")
        plt.close()


if __name__ == "__main__":
    main()