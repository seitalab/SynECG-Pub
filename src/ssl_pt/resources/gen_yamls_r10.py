import os
TEMPLATE = "./templates/template007.yaml"

def gen_yaml(
    pt_id: int, 
    exp_key: str, 
    d_lim: int, 
    dataset: str,
    arch_name: str
):
    """

    Args:
        pt_id (int): _description_
        exp_key (str): _description_
        arch_name (str): _description_
    """
    with open(TEMPLATE, "r") as f:
        template = f.read()

    # Replace values
    template = template.replace("<VAL01>", exp_key)
    template = template.replace("<VAL02>", dataset)
    template = template.replace("<VAL03>", str(d_lim))
    template = template.replace("<VAL04>", arch_name)

    # Write to file
    savename = os.path.join(
        "./pretrain_yamls",
        f"pt{pt_id//100:02d}s",
        f"pt{pt_id:04}.yaml"
    )
    os.makedirs(os.path.dirname(savename), exist_ok=True)
    with open(savename, "w") as f:
        f.write(template)

def main(exp_keys, arch_names, data_and_lims, pt_start: int=1):
    pt_id = pt_start
    for exp_key in exp_keys:
        for arch_name in arch_names:
            for dataset, d_lim in data_and_lims:
                gen_yaml(
                    pt_id, 
                    exp_key, 
                    d_lim, 
                    dataset,
                    arch_name
                )
                pt_id += 1
    print("Done")

if __name__ == "__main__":

    exp_keys = ["pt_syn02"]
    arch_names = ["transformer"]
    data_and_lims = [
        ("syn_ecg-04", 5_000),
        ("PTBXL-NORM", "null"),
    ]

    main(
        exp_keys, 
        arch_names, 
        data_and_lims, 
        pt_start=401
    )
