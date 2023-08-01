from typing import Optional
import re
from pathlib import Path

from pyscf import gto
import numpy as np
import scipy.special
import torch


# Atomic Units
AU = 1.0
Bohr = BOHR = 1.0
Hartree = HARTREE = 1.0

# energy
eV = EV = 27.211386024367243
mHartree = MHARTREE = Hartree * 1000
meV = MEV = eV * 1000
kcal_per_mol = KCAL_PER_MOL = 627.5094738898777  # although not a unit of energy, it is often used to represent energy

# length
Angstrom = ANGSTROM = 0.5291772105638411
# dipole
Debye = DEBYE = 2.5417464157449032
mDebye = MDEBYE = Debye * 1000

PROP_UNIT = None
LEN_UNIT = "Angstrom"

unit_set = {
    "AU", "BOHR", "HARTREE", "EV", "MHARTREE", "MEV", "KCAL_PER_MOL", "ANGSTROM", "DEBYE", "MDEBYE",
}

def eval_unit(unit: str):
    # check if the unit is valid and safe
    split_unit = re.split(r"[+ | \- | * | / | ^ | ( | )]", unit)
    for u in split_unit:
        if u == '':
            continue
        elif u in unit_set:
            continue
        elif u.isdigit():
            continue
        else:
            raise ValueError(f"Invalid unit expression: {u}")
    unit = unit.replace('^', '**')

    return eval(unit)


def unit_conversion(unit_in: Optional[str], unit_out: Optional[str]):
    if unit_in is None or unit_out is None:
        return 1.
    unit_in = unit_in.upper()
    unit_out = unit_out.upper()
    if unit_in == unit_out:
        return 1.
    value_in = eval_unit(unit_in)
    value_out = eval_unit(unit_out)

    return value_out / value_in

def set_default_unit(prop_unit: str, len_unit: str):
    global PROP_UNIT, LEN_UNIT
    PROP_UNIT = prop_unit
    LEN_UNIT = len_unit

def get_default_unit():
    return PROP_UNIT, LEN_UNIT


THIS_FOLDER = Path(__file__).parent
BASIS_FOLDER = THIS_FOLDER / "basis"
PRE_FOLDER = THIS_FOLDER / "pre_computed"

# periodic table of elements
# atomic numbers to element symbols
ELEMENTS_LIST = ['d',
  'H' ,                                                                                                 'He',
  'Li', 'Be',                                                             'B' , 'C' , 'N' , 'O' , 'F' , 'Ne',
  'Na', 'Mg',                                                             'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar',
  'K' , 'Ca', 'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
  'Rb', 'Sr', 'Y' , 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I' , 'Xe',
]
# element symbols to atomic numbers
ELEMENTS_DICT = {e: i for i, e in enumerate(ELEMENTS_LIST)}
# ground state multiplicities for each element
ATOM_MULT = [-1,
  2,                                                 1,
  2, 1,                               2, 3, 4, 3, 2, 1,
  2, 1,                               2, 3, 4, 3, 2, 1,
  2, 1, 2, 3, 4, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1,
  2, 1, 2, 3, 6, 7, 6, 5, 4, 1, 2, 1, 2, 3, 4, 3, 2, 1,
]
# atomic masses
ATOM_MASS = torch.Tensor([0.0,
    1.0080, 4.0026, 6.94,   9.0122, 10.81,  12.011, 14.007, 15.999, 18.998, 20.180,
    22.990, 24.305, 26.982, 28.085, 30.974, 32.06,  35.45,  39.95,  39.098, 40.078,
    44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.38,
    69.723, 72.630, 74.922, 78.971, 79.904, 83.798, 85.468, 87.62,  88.906, 91.224,
    92.906, 95.95,  98.,    101.07, 102.91, 106.42, 107.87, 112.41, 114.82, 118.71,
    121.76, 127.60, 126.90, 131.29,
])
# atomic energies at CCSD(T)/AHGBS-5 in Hartree
CC_ATOM_ENERGY = torch.DoubleTensor([0.0,
    -0.5,                -2.8789743710420823, -7.4735853872549045,  -14.66086539355887,  -24.618489920509003,
    -37.77873539152293,  -54.493372218958974, -74.93408104902329,   -99.565762667769,    -128.73876519337327,
    -162.03893799957353, -199.8197982721131,  -242.0738343479831,   -289.0419016677005,  -340.895954021737,
    -397.68939305632034, -459.6724933115134,  -527.0142415492794,   -599.7209115836976,  -677.3620106765908,
    -760.3512159960259,  -849.018198712152,   -943.5067439896815,   -1043.9712203798786, -1150.4062812979296,
    -1263.0422058284594, -1382.0962724844153, -1507.6115255825416,  -1639.7979759854961, -1778.6792007055597,
    -1924.0713566704283, -2076.157840716771,  -2235.0267459021884,  -2400.6631576850277, -2573.243483522314,
    -2752.8629931685527, -2939.160652158978,  -3132.3723325850056,  -3332.4745615309844, -3539.8002937779042,
    -3754.349504498755,  -3976.2813890211232, -4205.513231496394,   -4442.262920922795,  -4686.600084500385,
    -4938.6654926371875, -5198.411833925291,  -5465.847416806276,   -5740.861680559743,  -6023.604995076103,
    -6314.13783680609,   -6612.428613454331,  -6918.614467921217,   -7232.758318265938,
])
# atomic energies at PM7 in Hartree
PM7_ATOM_ENERGY = torch.DoubleTensor([0.0,
    0.830298220756e-01, 0.222044604925e-15,  0.612102311989e-01, 0.122643566599,     0.216251715014,
    0.272330549585,     0.180076962392,      0.949133079920e-01, 0.301031311468e-01, 0.177635683940e-14,
    0.408758768615e-01, 0.557760502984e-01,  0.123739226582,     0.172566429745,     0.120428460601,
    0.105259416178,     0.458812887577e-01, -0.177635683940e-14, 0.341349427826e-01, 0.678874212203e-01,
    0.380624526905,     0.140549567169,      0.195354155026,     0.151392136524,     0.179674918114,
    0.155308997006,     0.202239477529,      0.148434688775,     0.128603635974,     0.496725567943e-01,
    0.104221533986,     0.142627328620,      0.115217383902,     0.865325580344e-01, 0.384348875065e-01,
   -0.177635683940e-14, 0.312345881671e-01,  0.623098161905e-01, 0.119786735363,     0.234002299098,
    0.433294829034,     0.250673506055,      0.215696611424,     0.265452633300,     0.219696230385,
    0.143424129339,     0.108524257866,      0.425810303992e-01, 0.924288833516e-01, 0.115058023758,
    0.100715610825,     0.748992675436e-01,  0.383116673137e-01, 0.00000000000,
])
# atomic energies at GFN1-xTB in Hartree
GFN2_ATOM_ENERGY = torch.DoubleTensor([0.0,
    -0.393482763936, -1.743126632946, -0.180071686575, -0.569105981616, -0.952436614164,
    -1.793296371365, -2.605824161279, -3.767606950376, -4.619339964238, -5.932215052758,
    -0.167096749822, -0.465974663792, -0.905328611479, -1.569609938455, -2.374178794732,
    -3.146456870402, -4.482525134961, -4.279043267590, -0.165752239061, -0.371646352489,
    -0.854183293246, -1.365500218647, -1.715748172741, -1.660416970851, -2.435435377098,
    -2.787494867799, -3.427209889079, -4.521799939018, -3.748006130985, -0.527521402296,
    -1.081111835714, -1.808089637246, -2.235797655243, -3.118622050579, -4.048339371234,
    -4.271855540848, -0.159998948675, -0.462430853001, -1.194852897131, -1.309096670199,
    -1.657343281017, -1.731014633087, -2.361087129907, -2.849205104045, -3.745587067752,
    -4.409845299930, -3.821738210271, -0.533037255301, -1.125937778890, -2.011082469458,
    -2.160600494682, -3.007276817897, -3.779630263390, -3.883588498190,
])
# specializations for qm9 by Z.R. Qiao
QM9_ATOM_ENERGY = CC_ATOM_ENERGY.clone()
QM9_ATOM_ENERGY[[1, 6, 7, 8, 9]] = torch.DoubleTensor(
    [-0.6038066, -38.0740441, -54.7491437, -75.2252374, -99.8658573]
)


def gen_int2c1e(basisname="gfn2-xtb"):
    """
    Projection of atomic orbitals onto auxiliary basis.
    """
    int2c1e_dict = {}
    if (BASIS_FOLDER / f"{basisname}.dat").exists():
        basis = str(BASIS_FOLDER / f"{basisname}.dat")
    else:
        basis = basisname
    orbaux = str(BASIS_FOLDER / "orbaux.dat")
    savefile = PRE_FOLDER / f"{basisname}_embedding.pt"
    aux = gto.M(atom="X 0 0 0", basis={'X': orbaux})
    nao_aux = aux.nao
    ao_loc_nr = aux.ao_loc_nr()[:-1]

    for atom, mult in zip(ELEMENTS_LIST[1:], ATOM_MULT[1:]):
        mol = gto.M(
            atom=f"X 0 0 0; {atom} 0 0 0",
            basis={'X': orbaux, atom: basis},
            spin=mult - 1,
        )
        ovlp = mol.intor("int1e_ovlp")
        projection = ovlp[:nao_aux, nao_aux:]
        embedding = np.linalg.norm(projection, axis=-1) # / projection.shape[-1] 
        int2c1e_dict[atom] = torch.from_numpy(embedding[ao_loc_nr])
    torch.save(int2c1e_dict, savefile)


def get_embedding_tensor(basisname="gfn2-xtb") -> torch.Tensor:
    """
    Get embedding of atoms in a basis.

    Args:
        `basisname`: name of the basis
    Returns:
        a tensor of shape ``(n_atoms, n_aux)``
    """
    if not (PRE_FOLDER / f"{basisname}_embedding.pt").exists():
        gen_int2c1e(basisname)
    embed_dict = torch.load(PRE_FOLDER / f"{basisname}_embedding.pt")
    embed_tenor = torch.stack([embed_dict[atom] for atom in ELEMENTS_LIST[1:]])
    embed_tenor =  torch.cat([torch.zeros(1, embed_tenor.shape[-1]), embed_tenor])
    return embed_tenor.to(torch.get_default_dtype())


def get_atomic_energy(atom_ref: str = None) -> torch.Tensor:
    """
    Get the shift of the atomic energies for each element.

    Args:
        `atom_ref`: type of element shifts
    Returns:
        a tensor of shape ``(n_atoms,)``
    """
    if atom_ref is None:
        return torch.zeros(len(ELEMENTS_LIST), dtype=torch.float64)
    else:
        atomic_energy = globals().get(f"{atom_ref.upper()}_ATOM_ENERGY")
        return atomic_energy * unit_conversion("Hartree", PROP_UNIT)


def calc_cgto_norm(cgto: list):
    """cgto: [l, [exp1, coeff1], [exp2, coeff2], ...]"""
    intor = 0.
    l = cgto[0]
    for exp1, coeff1 in cgto[1:]:
        for exp2, coeff2 in cgto[1:]:
            intor += coeff1 * coeff2 * scipy.special.gamma(l + 0.5) / (2 * (exp1 + exp2)**(l + 0.5))
    return 1 / np.sqrt(intor)


def centrization(at_no: torch.Tensor, coords: torch.Tensor):
    """Centrization of the coordinates of atoms."""
    assert at_no.shape[0] == coords.shape[0]
    masses = ATOM_MASS[at_no]
    centroid = torch.sum(masses * coords, dim=0) / torch.sum(masses)
    return coords - centroid


if __name__ == "__main__":
    gen_int2c1e("pm7")
    gen_int2c1e("gfn2-xtb")