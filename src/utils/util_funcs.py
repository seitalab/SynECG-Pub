
def update_clf_mode(params, realdataset="g12ec"):

    if params.neg_dataset.find("-LEAD") != -1:
        if realdataset != "ptbxl":
            raise
        else:
            return params

    if params.neg_dataset.endswith("-MultiLead"):
        if realdataset != "ptbxl":
            raise
        else:
            return params

    # Select positive dataset for evaluation.
    if params.target_dx == "af":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-AFIB"
        elif realdataset == "g12ec":
            params.pos_dataset = "G12EC-Afib"
        elif realdataset == "cpsc":
            params.pos_dataset = "CPSC-AF"
        else:
            raise
    elif params.target_dx == "pvc":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-PVC"
        elif realdataset == "g12ec":
            params.pos_dataset = "G12EC-VPB"
        elif realdataset == "cpsc":
            params.pos_dataset = "CPSC-PVC"
        else:
            raise
    elif params.target_dx == "vf":
        params.pos_dataset = "cardially"

    elif params.target_dx == "aflt":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-AFLT"
        else:
            raise

    elif params.target_dx == "pac":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-PAC"
        elif realdataset == "g12ec":
            params.pos_dataset = "G12EC-PAC"
        elif realdataset == "cpsc":
            params.pos_dataset = "CPSC-PAC"
        else:
            raise

    elif params.target_dx == "irbbb":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-IRBBB"
        elif realdataset == "g12ec":
            params.pos_dataset = "G12EC-IRBBB"
        else:
            raise

    elif params.target_dx == "crbbb":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-CRBBB"
        else:
            raise

    elif params.target_dx == "std":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-STD_"
        elif realdataset == "cpsc":
            params.pos_dataset = "CPSC-STD"
        else:
            raise

    elif params.target_dx == "wpw":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-WPW"
        else:
            raise

    elif params.target_dx == "3avb":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-3AVB"
        else:
            raise

    elif params.target_dx == "asmi":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-ASMI"
        else:
            raise

    elif params.target_dx == "imi":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-IMI"
        else:
            raise

    elif params.target_dx == "irbbb-crbbb":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-CRBBB"
        else:
            raise

    elif params.target_dx == "lvh":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-LVH"
        elif realdataset == "g12ec":
            params.pos_dataset = "G12EC-LVH"
        else:
            raise

    elif params.target_dx == "lafb":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-LAFB"
        else:
            raise

    elif params.target_dx == "isc":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-ISC_"
        else:
            raise

    elif params.target_dx == "iavb":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-1AVB"
        elif realdataset == "g12ec":
            params.pos_dataset = "G12EC-IAVB"
        elif realdataset == "cpsc":
            params.pos_dataset = "CPSC-IAVB"
        else:
            raise

    elif params.target_dx == "abqrs":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-ABQRS"
        else:
            raise

    elif params.target_dx == "rbbb":
        if realdataset == "g12ec":
            params.pos_dataset = "G12EC-RBBB"
        elif realdataset == "cpsc":
            params.pos_dataset = "CPSC-RBBB"
        else:
            raise

    else:
        raise NotImplementedError(f"{params.target_dx} is not implemented")
    
    # Negative dataset.
    if params.target_dx == "irbbb-crbbb":
        params.neg_dataset = "PTBXL-IRBBB"
    else:
        if realdataset == "ptbxl":
            params.neg_dataset = "PTBXL-NORM"
        elif realdataset == "g12ec":
            params.neg_dataset = "G12EC-NormalSinus"
        elif realdataset == "cpsc":
            params.neg_dataset = "CPSC-NORM"
        else:
            raise NotImplementedError(f"{realdataset} is invalid.")
    
    return params