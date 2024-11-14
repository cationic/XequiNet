from typing import Dict, Optional
import torch
from ase.stress import full_3x3_to_voigt_6_stress
from ase.calculators.calculator import Calculator, all_changes
from ase import units
from ..utils import radius_graph_pbc

class XeqCalculator(Calculator):
    """
    ASE calculator for XequiNet models.
    Supports both MD and geometry optimization model types.
    """
    implemented_properties = ["energy"]  # Common properties for both types
    default_parameters = {
        "cutoff": 5.0,
        "max_edges": 100,
        "charge": 0,
        "spin": 0,
    }

    def __init__(
        self,
        model_type: str = "geometry",
        ckpt_file: Optional[str] = None,
        model: Optional[torch.jit.ScriptModule] = None,
        **kwargs
    ) -> None:
        """
        Initialize XeqCalculator.
        
        Args:
            model_type: Type of the model ('md' or 'geometry'). Defaults to 'geometry'.
            ckpt_file: Path to the model checkpoint file. Defaults to None.
            model: Pre-loaded model instance. Defaults to None.
            **kwargs: Additional parameters for the Calculator.
        """
        if model_type not in ["md", "geometry"]:
            raise ValueError("model_type must be either 'md' or 'geometry'")
        
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Add additional properties
        if self.model_type == "md":
            self.implemented_properties += ["energies", "stress", "stresses", "forces"]
        else:
            self.implemented_properties += ["gradient"]
        
        # Initialize model
        self.model = None
        if model is not None:
            self.model = model.to(self.device)
        elif ckpt_file is not None:
            self.model = torch.jit.load(ckpt_file, map_location=self.device)
        else:
            raise ValueError("Either ckpt_file or model must be provided")
            
        super().__init__(**kwargs)

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ) -> None:
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Common inputs for both model types
        positions = self.atoms.positions
        at_no = torch.from_numpy(self.atoms.numbers).to(torch.long).to(self.device)
        
        # Convert coordinates from Angstrom to Bohr if needed
        if self.model_type == "geometry":
            coord = torch.from_numpy(positions / units.Bohr).to(torch.get_default_dtype()).to(self.device) # Angstrom to Bohr
        elif self.model_type == "md":
            coord = torch.from_numpy(positions).to(torch.get_default_dtype()).to(self.device)
        
        model_inputs = {
            "at_no": at_no,
            "coord": coord,
            "charge": self.parameters.charge,
            "spin": self.parameters.spin,
        }
        
        # Add MD-specific inputs
        if self.model_type == "md":
            edge_index, shifts = radius_graph_pbc(
                positions=positions,
                pbc=self.atoms.pbc,
                cell=self.atoms.cell,
                cutoff=self.parameters.cutoff,
                max_num_neighbors=self.parameters.max_edges,
            )
            edge_index = torch.from_numpy(edge_index).to(torch.long).to(self.device)
            shifts = torch.from_numpy(shifts).to(torch.get_default_dtype()).to(self.device)
            model_inputs.update({
                "edge_index": edge_index,
                "shifts": shifts,
            })

        # Run model
        model_res: Dict[str, torch.Tensor] = self.model(**model_inputs)
        
        if self.model_type == "geometry":
            self.results["energy"] = model_res["energy"].item() * units.Hartree  # Hartree to eV
            self.results["forces"] = - model_res["gradient"].detach().cpu().numpy() * units.Hartree / units.Bohr  # Convert Ha/Bohr to eV/Angstrom
        elif self.model_type == "md":
            self.results["energy"] = model_res["energy"].item()
            self.results["forces"] = model_res["forces"].detach().cpu().numpy()
            self.results["energies"] = model_res["energies"].detach().cpu().numpy() 
            
            # Process stress and stresses if cell is 3D
            if self.atoms.cell.rank == 3:
                virials = model_res["virials"].detach().cpu().numpy() 
                virial = model_res["virial"].detach().cpu().numpy()
                self.results["stress"] = full_3x3_to_voigt_6_stress(virial) / self.atoms.get_volume()
                self.results["stresses"] = full_3x3_to_voigt_6_stress(virials) / self.atoms.get_volume()
