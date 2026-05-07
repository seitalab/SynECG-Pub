import os
from glob import glob

import yaml
import pandas as pd

from exp01_hps import (
    generate_exp_yaml, 
    get_all_combinations, 
    load_template,
    replace_value,
    save_exp_yaml    
)
from exp02_clf import get_df_hps_result_path

# Use previously conducted hyperparameter search results.
pt_id_table = {
    "pt0301": "pt0006", # PTBXL-Normal (n=4600)
    "pt0302": "pt0006", # PTBXL-All (n=5000)
    "pt0105": "pt0006"  # SimECG-N (n=5000)
}

def load_df_hps_result_path(df, settings):
    """
    Args:
        df (pd.DataFrame): _description_
        settings (List): [ft_target, dataset, target_dx]
    Returns:
        str
    """

    ft_target, dataset, target_dx = settings
    ft_target_prev = pt_id_table[ft_target]
    df = df[
        (df.finetune_target == ft_target_prev) &
        (df.dataset == dataset) &
        (df.target_dx == target_dx)
    ]
    return df["result_path"].values[0]


def generate_exp_yaml(
    exp_id_start, 
    template_id, 
    val_replace_dict,
    df_hps_result_path
):
    """
    Args:
        exp_id (_description_): _description_
        template_id (_description_): _description_

    Returns:
        str: _description_
    """
    template = load_template(template_id)
    
    all_combinations = get_all_combinations(val_replace_dict)
    for n_proc, comb in enumerate(all_combinations):
        exp_yaml = template
        exp_id = exp_id_start + n_proc

        for key, val in comb.items():
            exp_yaml = replace_value(exp_yaml, key, val)

        # Insert hps_path.
        settings = [val for val in comb.values()][:3] # [pt_id, dataset, dx, lim_val] -> [pt_id, dataset, dx]
        hps_path = load_df_hps_result_path(df_hps_result_path, settings)
        if hps_path == "N/A":
            print(exp_id, settings)
            continue
        exp_yaml = replace_value(exp_yaml, "VAL05", hps_path)
        save_exp_yaml(exp_yaml, exp_id)
    return exp_id, exp_yaml

if __name__ == "__main__":
    exp_yaml_start = 8301
    template_id = 6

    # Load hyperparameter search results.
    exp_ids = list(range(1, 105))
    exp_ids += list(range(417, 521)) # Add PTBXL-pt0006 results.
    target_keys = ["target_dx", "dataset", "finetune_target"]
    df_hps_result_path = get_df_hps_result_path(
        exp_ids,
        target_keys
    )

    val_replace_dict = {}
    val_replace_dict["VAL01"] = [
        "pt0301", "pt0302", 
        "pt0105"
    ]

    # PTBXL
    val_replace_dict["VAL02"] = ["ptbxl"]
    val_replace_dict["VAL03"] = [
        "af", "asmi", "abqrs", "crbbb", 
        "imi", "irbbb", "isc", "lafb", 
        "lvh", "pac", "pvc", "std", "1avb", 
    ]
    val_replace_dict["VAL04"] = ["null"]
    print(val_replace_dict)

    last_id, sample_yaml = generate_exp_yaml(
        exp_yaml_start, 
        template_id, 
        val_replace_dict,
        df_hps_result_path
    )

    # G12EC
    val_replace_dict["VAL02"] = ["g12ec"]
    val_replace_dict["VAL03"] = [
        "af", "pvc", "lvh", "irbbb", "iavb", "pac", "rbbb"]
    print(val_replace_dict)

    last_id, sample_yaml = generate_exp_yaml(
        last_id + 1, 
        template_id, 
        val_replace_dict,
        df_hps_result_path
    )

    # CPSC
    val_replace_dict["VAL02"] = ["cpsc"]
    val_replace_dict["VAL03"] = [
        "af", "iavb", "pac", "pvc", "std", "rbbb"]
    print(val_replace_dict)

    last_id, sample_yaml = generate_exp_yaml(
        last_id + 1, 
        template_id, 
        val_replace_dict,
        df_hps_result_path
    )

    print("-"*80)
    print("Sample YAML:")
    print(sample_yaml)
    print("-"*80)
    print(f"Last ID: {last_id}")
