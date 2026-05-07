# Exp01 experiment index

Hyperparameter search + multiseed classification

> Experiment definitions are centrally managed through `exp_builder.py`.
> Legacy `expXX.py` runners were moved to `legacy_exp_generators/`.

## Exp01

- 1 - 104
- PTBXL
- pt0001-pt0010 (excluding pt0004, pt0009)
- af, asmi, abqrs, crbbb, imi, irbbb, isc, lafb, lvh, pac, pvc, std, 1avb

## Exp01b

- 401 - 416
- PTBXL
- pt0001-pt0010 (excluding pt0004, pt0009)
- aflt, wpw

## Exp01c

- 417 - 472
- G12EC
- pt0001-pt0010 (excluding pt0004, pt0009)
- af, pvc, lvh, irbbb, iavb, pac, rbbb

## Exp01d

- 473 - 520
- CPSC
- pt0001-pt0010 (excluding pt0004, pt0009)
- af, iavb, pac, pvc, std, rbbb

## Exp01e

- 1001 - 1056
- PTBXL, G12EC, CPSC
- pt0011, pt0016
- PTBXL-15dx, G12EC-7dx, CPSC-6dx

## Exp01f

- 801 - 856
- PTBXL, G12EC, CPSC
- pt0004, pt0009
- PTBXL-15dx, G12EC-7dx, CPSC-6dx

## Exp01g

- 1101 - 1268
- PTBXL, G12EC, CPSC
- pt0012, pt0014, pt0015, pt0017, pt0019, pt0020
- PTBXL-15dx, G12EC-7dx, CPSC-6dx

## Exp01h

- 1301 - 1356
- PTBXL, G12EC, CPSC
- pt0013, pt0018
- PTBXL-15dx, G12EC-7dx, CPSC-6dx

## Exp01i

- 1401 - 1428
- PTBXL, G12EC, CPSC
- pt0021
- PTBXL-15dx, G12EC-7dx, CPSC-6dx

## Exp01j

- 1501 - 1526
- PTBXL, G12EC, CPSC
- pt-extra01
- PTBXL-13dx, G12EC-7dx, CPSC-6dx

# Exp02

Multiseed classification (run when Exp01 fails after hyperparameter search).

## Exp02

- 201 - 304
- PTBXL
- pt0001-pt0010 (excluding pt0004, pt0009)
- af, asmi, abqrs, crbbb, imi, irbbb, isc, lafb, lvh, pac, pvc, std, 1avb
- re-execute 1 - 104

## Exp02b

- 601 - 720
- PTBXL, G12EC, CPSC
- pt0001-pt0010 (excluding pt0004, pt0009)
- PTBXL-15dx, G12EC-7dx, CPSC-6dx
- re-execute 401 - 520

# Exp03

Pretraining progress vs downstream task performance.

## Exp03

- 2001-2091, 2092-2140, 2141-2182
- pt0006 progress (7 steps)
- PTBXL-13dx, G12EC-7dx, CPSC-6dx

# Exp04

Limit the amount of data used during finetuning.

## Exp04

- 3001-3130
- PTBXL
- pt0006 (data limit)
- PTBXL-13dx

## Exp04b

- 3201-3330
- PTBXL
- pt0001 (data limit)
- PTBXL-13dx

# Exp05

Other lead experiment.

## Exp05

- 4001-4143
- PTBXL
- pt0006 (other leads)
- PTBXL-13dx

# Exp06

Demographics.

## Exp06

- 5001-5026
- PTBXL
- pt0001, pt0006
- PTBXL-13dx

## Exp07: VAE/GAN/McSherry data limit

- 5101-5490
- PTBXL
- pt0021, pt0016, pt0011 (data limit)
- PTBXL-13dx

## Exp08: Pretrain with reduced data

- 5501-5604
- PTBXL
- pt0101-0106,111,112
- PTBXL-13dx

## Exp09: Equal compute

- 6001-7040
- PTBXL
- pt0101-0106,111,112
- PTBXL-13dx

# Exp14

Compare PTBXL-Normal and PTBXL-All.

# Exp15, 16

Versus diffusion synthetic data (100k setting) for ICML26 rebuttal.
- 8201 - 8282
