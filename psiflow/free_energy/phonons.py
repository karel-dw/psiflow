import xml.etree.ElementTree as ET
from typing import Optional, Union

import numpy as np
import parsl
from ase.units import Bohr, Ha, J, _c, _hplanck, _k, kB, second
from parsl.app.app import bash_app, python_app, join_app
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset
from psiflow.geometry import Geometry, mass_weight
from psiflow.hamiltonians import Hamiltonian, MACEHamiltonian, MixtureHamiltonian
from psiflow.sampling.sampling import (
    setup_sockets,
    make_server_command,
    make_driver_commands,
    make_wait_for_sockets_command,
)
from psiflow.sampling.optimize import setup_forces
from psiflow.utils.apps import multiply
from psiflow.utils.io import load_numpy, save_xml
from psiflow.utils.parse import format_env_vars


def _compute_frequencies(hessian: np.ndarray, geometry: Geometry) -> np.ndarray:
    assert hessian.shape[0] == hessian.shape[1]
    assert len(geometry) * 3 == hessian.shape[0]
    return np.sqrt(np.linalg.eigvalsh(mass_weight(hessian, geometry))) / (2 * np.pi)


compute_frequencies = python_app(_compute_frequencies, executors=["default_threads"])


def _harmonic_free_energy(
    frequencies: Union[float, np.ndarray],
    temperature: float,
    quantum: bool = False,
    threshold: float = 1,  # in invcm
) -> float:
    if isinstance(frequencies, float):
        frequencies = np.array([frequencies], dtype=float)

    threshold_ = threshold / second * (100 * _c)  # from invcm to ASE
    frequencies = frequencies[np.abs(frequencies) > threshold_]

    # _hplanck in J s
    # _k in J / K
    if quantum:
        arg = (-1.0) * _hplanck * frequencies * second / (_k * temperature)
        F = kB * temperature * np.sum(np.log(1 - np.exp(arg)))
        F += _hplanck * J * second * np.sum(frequencies) / 2
    else:
        constant = kB * temperature * np.log(_hplanck)
        actual = np.log(frequencies / (kB * temperature))
        F = len(frequencies) * constant + kB * temperature * np.sum(actual)
    F /= kB * temperature
    return F


harmonic_free_energy = python_app(_harmonic_free_energy, executors=["default_threads"])


def setup_motion(
    asr: str,
    pos_shift: float,
    energy_shift: float,
) -> ET.Element:
    motion = ET.Element("motion", mode="vibrations")
    vibrations = ET.Element("vibrations", mode="fd")
    pos = ET.Element("pos_shift")
    pos.text = " {} ".format(pos_shift)
    vibrations.append(pos)
    energy = ET.Element("energy_shift")
    energy.text = " {} ".format(energy_shift)
    vibrations.append(energy)
    prefix = ET.Element("prefix")
    prefix.text = " output "
    vibrations.append(prefix)
    asr_ = ET.Element("asr")
    asr_.text = " {} ".format(asr)
    vibrations.append(asr_)
    motion.append(vibrations)
    return motion


def _execute_ipi(
    driver_kwargs: list[dict],
    command_server: str,
    env_vars: dict = {},
    bash_template: str = "",
    stdout: str = parsl.AUTO_LOGNAME,
    stderr: str = parsl.AUTO_LOGNAME,
    inputs: list = [],
    outputs: list = [],
    parsl_resource_specification: Optional[dict] = None,
) -> str:
    file_xml, file_xyz_in, *files_in = inputs
    command_start = make_server_command(
        command_server, file_xml, file_xyz_in, outputs[0], [], []
    )
    command_wait = make_wait_for_sockets_command(
        set(d["address"] for d in driver_kwargs)
    )
    commands_driver = make_driver_commands(driver_kwargs, file_xyz_in, files_in)
    command_list = [
        command_start,
        command_wait,
        *commands_driver,
        "wait",
        f"cp i-pi.output_full.hess {outputs[0]}",
    ]
    commands, env = "\n".join(command_list), format_env_vars(env_vars)
    return bash_template.format(commands=commands, env=env)


execute_ipi = bash_app(_execute_ipi, executors=["ModelEvaluation"])


def compute_harmonic(
    state: Union[Geometry, AppFuture],
    hamiltonian: Hamiltonian,
    mode: str = "fd",
    asr: str = "crystal",
    pos_shift: float = 0.01,
    energy_shift: float = 0.00095,
) -> AppFuture:
    components, force_xml = setup_forces(hamiltonian)
    sockets = setup_sockets(components)

    initialize = ET.Element("initialize", nbeads="1")
    start = ET.Element("file", mode="ase", cell_units="angstrom")
    start.text = " start_0.xyz "
    initialize.append(start)
    motion = setup_motion(asr, pos_shift, energy_shift)
    if mode != "fd":
        raise NotImplementedError

    system = ET.Element("system")
    system.append(initialize)
    system.append(motion)
    system.append(force_xml)

    simulation = ET.Element("simulation", mode="static")
    for socket in sockets:
        simulation.append(socket)
    simulation.append(system)
    total_steps = ET.Element("total_steps")
    total_steps.text = " {} ".format(1000000)
    simulation.append(total_steps)

    context = psiflow.context()
    definition = context.definitions["ModelEvaluation"]
    input_future = save_xml(
        simulation,
        outputs=[context.new_file("input_", ".xml")],
    ).outputs[0]
    inputs = [input_future, Dataset([state]).extxyz]

    driver_kwargs = []
    for i, comp in enumerate(components):
        inputs.append(comp.hamiltonian.serialize_function(dtype="float64"))
        kwargs = {"idx": i, "address": comp.address}
        if isinstance(comp.hamiltonian, MACEHamiltonian):
            kwargs |= definition.get_driver_resources(1, 1)[0]
        driver_kwargs.append(kwargs)

    result = execute_ipi(
        driver_kwargs,
        definition.server_command(),
        env_vars=definition.env_vars,
        bash_template=context.bash_template,
        inputs=inputs,
        outputs=[context.new_file("hess_", ".txt")],
        parsl_resource_specification=definition.wq_resources(1),
    )
    return multiply(load_numpy(inputs=[result.outputs[0]]), Ha / Bohr**2)


@python_app
def ehessian_rows(
    forces: AppFuture | list[np.ndarray],
    stresses: AppFuture | list[np.ndarray],
    pert_geos: list[Geometry],
    h_ref: np.ndarray,
    delta_dh: float,
    num_atoms: int,
) -> np.ndarray:

    import numpy as np

    # extract force pairs, stress pairs, and h pairs for each row of ehessian
    force_pairs = []
    stress_pairs = []
    h_pairs = []
    for i in range(0, len(forces), 2):
        force_pairs.append((forces[i], forces[i + 1]))
        stress_pairs.append((stresses[i], stresses[i + 1]))
        h_pairs.append((pert_geos[i].cell, pert_geos[i + 1].cell))

    # compute rows of ehessian
    rows = []
    
    for force_pair, stress_pair, h_pair in zip(
        force_pairs, stress_pairs, h_pairs):

        row = np.zeros(3 * num_atoms + 9)

        # compute first 3N elements of row
        dE_dd_pair = []
        for force, h in zip(force_pair, h_pair):
            dE_dr = (-1.0) * force
            dE_dd = dE_dr @ (np.linalg.inv(h_ref) @ h).T
            dE_dd_pair.append(dE_dd)
        row[:3 * num_atoms] = ((dE_dd_pair[0] - dE_dd_pair[1]) / (2 * delta_dh)).flatten()
        
        # compute last 9 elements of row
        dE_dh_pair = []
        for stress, h in zip(stress_pair, h_pair):
            volume = np.linalg.det(h)
            dE_dh = volume * (stress @ np.linalg.inv(h)).T
            dE_dh_pair.append(dE_dh)
        row[3 * num_atoms:] = ((dE_dh_pair[0] - dE_dh_pair[1]) / (2 * delta_dh)).flatten()

        rows.append(row)
    
    return np.array(rows)


@join_app
def compute_ehessian(
    geometry_ref: Geometry,
    hamiltonian: Hamiltonian,
    delta_dh: float = 1e-5,
    ehessian_coordinates: str | None = None,
) -> np.ndarray:
    
    import numpy as np
    
    """See section 2.1 of the Supplementary Information:
    https://arxiv.org/abs/2602.20738

    Compute the extended Hessian (ehessian) of a given potential energy
    surface around a given reference geometry.

    Generates finite-difference perturbations of atomic positions and cell
    degrees of freedom, evaluates forces and stresses, and assembles the 
    ehessian matrix.

    Args:
        geometry_ref: Reference Geometry object.
        hamiltonian: Hamiltonian used to compute energies, forces, and stresses.
        delta_dh: Finite difference step size [Angstrom].
        ehessian_coordinates: Coordinate type for ehessian ("dh" supported).
        check_ehessian: If True, verify and enforce symmetry of ehessian.

    Returns:
        np.ndarray: ehessian matrix of shape (3N + 9, 3N + 9).

    Raises:
        NotImplementedError: If unsupported ehessian_coordinates are requested.
    """

    # if isinstance(hamiltonian, MixtureHamiltonian):
    #    for h in hamiltonian.hamiltonians:
    #        if isinstance(h, MACEHamiltonian) and h.dtype != "float64":
    #            print(
    #                f"Warning: MACEhamiltonian dtype is {h.dtype}, and is set to "
    #                "float64, which is required for accurate ehessian calculation"
    #            )
    #            h.dtype = "float64"
    # elif isinstance(hamiltonian, MACEHamiltonian) and hamiltonian.dtype != "float64":
    #    print(
    #        f"Warning: MACEhamiltonian dtype is {hamiltonian.dtype}, and is set to "
    #        "float64, which is required for accurate ehessian calculation"
    #    )
    #    hamiltonian.dtype = "float64"
    
    if ehessian_coordinates != "dh":
        raise NotImplementedError(
            "Only 'dh' is supported for ehessian_coordinates, "
            f"but got {ehessian_coordinates}."
        )
    
    num_atoms = len(geometry_ref)
    h_ref = np.copy(geometry_ref.cell)
    d_ref = np.copy(geometry_ref.per_atom.positions)

    ## compute oppositely perturbed geometry pairs
    ## for each deformed atomic coordinate and for each cell coordinate
    pert_geos = []

    # compute first 3N x 2 perturbed geometries (for first 3N rows of ehessian)
    for i in range(num_atoms):
        for j in range(3):
            for direction in (+1, -1):
                d_pert = np.copy(d_ref)
                d_pert[i, j] += direction * delta_dh

                tmp_geo = geometry_ref.copy()
                tmp_geo.per_atom.positions = d_pert

                pert_geos.append(tmp_geo)

    # compute last 9 x 2 perturbed geometries (for last 9 rows of ehessian)
    for i in range(3):
        for j in range(3):
            for direction in (+1, -1):
                h_pert = np.copy(h_ref)
                h_pert[i, j] += direction * delta_dh
                d_pert = np.copy(d_ref) @ np.linalg.inv(h_ref) @ h_pert

                tmp_geo = geometry_ref.copy()
                tmp_geo.per_atom.positions = d_pert
                tmp_geo.cell = h_pert

                pert_geos.append(tmp_geo)

    ## compute forces and stresses via a single .compute call for all 2 x (3N + 9)
    ## perturbed geometries
    _energies, forces, stresses = hamiltonian.compute(pert_geos)

    ## compute (3N + 9) rows of ehessian
    ehessian = ehessian_rows(forces, stresses, pert_geos, h_ref, delta_dh, num_atoms)

    return ehessian