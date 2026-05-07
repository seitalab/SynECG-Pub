from gen_yamls_r04 import main

if __name__ == "__main__":

    exp_keys = ["pt_syn02"]
    ssl_names = ["mae"]
    datasets = ["PTBXL-ALL", "G12EC-ALL"]
    d_lims = ["null"]
    main(
        exp_keys, 
        ssl_names, 
        d_lims, 
        datasets, 
        pt_start=111
    )
