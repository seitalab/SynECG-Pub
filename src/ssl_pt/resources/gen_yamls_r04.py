import os
TEMPLATE = "./templates/template002.yaml"

def gen_yaml(
    pt_id: int, 
    exp_key: str, 
    d_lim: int, 
    dataset: str,
    ssl_name: str
):
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
    template = template.replace("<VAL03>", str(d_lim))
    template = template.replace("<VAL04>", dataset)

    # Write to file
    savename = os.path.join(
        "./pretrain_yamls",
        f"pt{pt_id//100:02d}s",
        f"pt{pt_id:04}.yaml"
    )
    os.makedirs(os.path.dirname(savename), exist_ok=True)
    with open(savename, "w") as f:
        f.write(template)

def main(exp_keys, ssl_names, d_lims, datasets, pt_start: int=1):
    pt_id = pt_start
    for exp_key in exp_keys:
        for ssl_name in ssl_names:
            for d_lim in d_lims:
                for dataset in datasets:
                    gen_yaml(
                        pt_id, 
                        exp_key, 
                        d_lim, 
                        dataset,
                        ssl_name
                    )
                    pt_id += 1
    print("Done")

if __name__ == "__main__":

    exp_keys = ["pt_syn02"]
    ssl_names = ["mae"]
    datasets = ["syn_ecg-04"]
    d_lims = [
        200_000, 100_000, 50_000, 20_000
    ]
    main(
        exp_keys, 
        ssl_names, 
        d_lims, 
        datasets, 
        pt_start=101
    )
