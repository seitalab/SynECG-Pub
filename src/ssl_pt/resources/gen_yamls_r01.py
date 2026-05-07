import os
TEMPLATE = "./templates/template001.yaml"

def gen_yaml(pt_id: int, exp_key: str, ssl_name: str):
    """

    Args:
        pt_id (int): _description_
        exp_key (str): _description_
        ssl_name (str): _description_
    """
    with open(TEMPLATE, "r") as f:
        template = f.read()

    # Replace values
    template = template.replace("<VAL01>", ssl_name)
    template = template.replace("<VAL02>", exp_key)

    # Write to file
    savename = os.path.join(
        "./pretrain_yamls",
        f"pt{pt_id//100:02d}s",
        f"pt{pt_id:04}.yaml"
    )
    os.makedirs(os.path.dirname(savename), exist_ok=True)
    with open(savename, "w") as f:
        f.write(template)

def main(exp_keys, ssl_names, pt_start: int=1):
    pt_id = pt_start
    for exp_key in exp_keys:
        for ssl_name in ssl_names:
            gen_yaml(pt_id, exp_key, ssl_name)
            pt_id += 1
    print("Done")

if __name__ == "__main__":

    exp_keys = ["pt_ptbxl01", "pt_syn01"]
    ssl_names = ["mae"]
    main(exp_keys, ssl_names, pt_start=1)
