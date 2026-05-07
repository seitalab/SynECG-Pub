import os
from itertools import product

def load_template(template_id: int):
    """
    Args:
        template_id (int): _description_

    Returns:
        str: _description_
    """
    template_file = f"./templates/template{template_id:03d}.yaml"
    with open(template_file, "r") as f:
        template = f.read()
    return template

def get_all_combinations(rep_dict):
    """
    
    rep_dict = {"a": [A1, A2], "b": [B1, B2, B3]}
    then,
    all_combinations = [
        {"a": "A1", "b": "B1"},
        {"a": "A1", "b": "B2"},
        {"a": "A1", "b": "B3"},
        {"a": "A2", "b": "B1"},
        {"a": "A2", "b": "B2"},
        {"a": "A2", "b": "B3"},
    ]

    Args:
        rep_dict (_description_):

    Returns:
        _description_: _description
    """
    all_combinations = []
    keys = list(rep_dict.keys())
    vals = list(rep_dict.values())
    for comb in product(*vals):
        comb_dict = {keys[i]: comb[i] for i in range(len(keys))}
        all_combinations.append(comb_dict)
    return all_combinations

def save_exp_yaml(exp_yaml, exp_id):
    """
    Args:
        exp_yaml (_description_): _description_
        exp_id (_description_): _description_
    """
    save_dir = f"exp{exp_id//100:02d}s"
    os.makedirs(save_dir, exist_ok=True)
    savename = os.path.join(save_dir, f"exp{exp_id:04d}.yaml")

    with open(savename, "w") as f:
        f.write(exp_yaml)

def replace_value(exp_yaml, key, val):
    """
    Args:
        exp_yaml (_description_): _description_
        key (_description_): _description_
        val (_description_): _description_

    Returns:
        str: _description_
    """
    return exp_yaml.replace(f"<{key}>", val)

def generate_exp_yaml(
    exp_id_start, 
    template_id, 
    val_replace_dict
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
        save_exp_yaml(exp_yaml, exp_id)
    return exp_id
    
if __name__ == "__main__":
    exp_yaml_start = 1
    template_id = 1
    val_replace_dict = {
        "VAL01": [
            "pt0001", "pt0002", "pt0003",
            "pt0005", "pt0006",
            "pt0007", "pt0008", 
            "pt0010"
        ],
        "VAL02": ["ptbxl"],
        "VAL03": [
            "af", "asmi", "abqrs", 
            "crbbb", "imi", "irbbb",
            "isc", "lafb",
            "lvh", "pac", "pvc",
            "std", "1avb"
        ],
    }

    generate_exp_yaml(
        exp_yaml_start, 
        template_id, 
        val_replace_dict
    )
    
