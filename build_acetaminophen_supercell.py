#!/usr/bin/env python3
"""
Build a LAMMPS data file (atom_style full) for acetaminophen (Form I, monoclinic P21/n)
using atom types/charges/topology read from a CGenFF .str file.

USAGE:
    python make_apap_full_from_str.py input.str output.data nx ny nz

NOTES:
- This script embeds the crystallographic geometry for APAP Form I (150 K) and applies P21/n symmetry
  to generate the primitive cell (4 molecules, 80 atoms). Then it tiles to an nx×ny×nz supercell.
- It reads the .str *residue* (ATOM lines: name→CGenFF type & charge; BOND lines; IMPR lines).
- ANGLES & DIHEDRALS are auto-generated from the bond graph (you’ll set coefficients later).
- No *Coeffs sections* are written. Load CHARMM/CGenFF parameters in your LAMMPS input.
- All numeric *type IDs* are mapped to human-readable keys (commented tables in the header).
"""

import sys, math, re
from collections import defaultdict, deque

# ------------------ CLI ------------------
if len(sys.argv) != 6:
    sys.exit("Usage: python build_acetaminophen_supercell.py acetaminophen_cgenff.str INPUT_FILENAME.data nx ny nz")

in_str = sys.argv[1]
out_data = sys.argv[2]

try:
    nx = int(sys.argv[3])
    ny = int(sys.argv[4])
    nz = int(sys.argv[5])
except ValueError:
    sys.exit("ERROR: nx, ny, nz must be integers (e.g., 1 1 1).")

if nx <= 0 or ny <= 0 or nz <= 0:
    sys.exit("ERROR: nx, ny, nz must all be positive integers.")


# ------------------ Parse .str ------------------
# We expect ONE residue block: ATOM <name> <cgenff_type> <charge>
# BOND lines: BOND A B
# Improper lines: IMPR a b c d
atom_types = {}   # name -> CGenFF type string
atom_chg   = {}   # name -> float
bonds_tpl  = []   # list of (name_i, name_j)
improp_tpl = []   # list of (a,b,c,d) as listed in .str

with open(in_str, "r") as f:
    lines = f.read().splitlines()

in_res = False
for ln in lines:
    s = ln.strip()
    if not s or s.startswith("!"):
        continue
    if s.upper().startswith("RESI "):
        in_res = True
        continue
    if in_res and s.upper().startswith("END"):
        in_res = False
        continue
    if not in_res:
        # later sections (BONDS/ANGLES/...) are empty in your .str; we don’t need them
        # because we’ll generate angles/dihedrals from BOND graph and read impropers above.
        pass

    if s.upper().startswith("ATOM "):
        # ATOM <name> <type> <charge> ...
        toks = s.split()
        # Some .str files have comments after; we only need first 4 tokens
        # e.g., ATOM O1 OG311 -0.533 ! ...
        if len(toks) < 4:
            sys.exit(f"ERROR parsing ATOM line: {ln}")
        _, name, typ, chg = toks[:4]
        atom_types[name] = typ
        try:
            atom_chg[name] = float(chg)
        except:
            sys.exit(f"ERROR parsing charge in line: {ln}")
        continue

    if s.upper().startswith("BOND "):
        # BOND A B
        toks = s.split()
        if len(toks) < 3:
            sys.exit(f"ERROR parsing BOND line: {ln}")
        _, a, b = toks[:3]
        bonds_tpl.append((a, b))
        continue

    if s.upper().startswith("IMPR "):
        # IMPR a b c d
        toks = s.split()
        if len(toks) < 5:
            sys.exit(f"ERROR parsing IMPR line: {ln}")
        _, a, b, c, d = toks[:5]
        improp_tpl.append((a, b, c, d))
        continue

if not atom_types or not atom_chg:
    sys.exit("ERROR: No ATOM records found in .str (cannot proceed).")
if not bonds_tpl:
    sys.exit("ERROR: No BOND records found in .str (cannot proceed).")

# ------------------ APAP Form I geometry (150 K) ------------------
# Lattice parameters (Å, degrees) for monoclinic, unique b, P21/n
a_lat = 7.0939
b_lat = 9.2625
c_lat = 11.6570
beta_deg = 97.672
beta = math.radians(beta_deg)

# Lattice vectors (triclinic representation)
ax, ay, az = a_lat, 0.0, 0.0
bx, by, bz = 0.0, b_lat, 0.0
cx = c_lat * math.cos(beta)
cy = 0.0
cz = c_lat * math.sin(beta)


# Asymmetric unit fractional coordinates (labels MUST match .str names)
asu = [
("O1",  "O",  0.1649,   0.92111,  0.22312),
("H1p", "H",  0.0520,   0.9570,   0.2410 ),
("C1",  "C",  0.1346,   0.8294,   0.12968),
("C2",  "C", -0.0365,   0.8277,   0.05496),
("H2",  "H", -0.1390,   0.8907,   0.0690 ),
("C3",  "C", -0.0585,   0.7355,  -0.03939),
("H3",  "H", -0.1790,   0.7345,  -0.0914 ),
("C4",  "C",  0.0888,   0.6438,  -0.06137),
("C5",  "C",  0.2581,   0.6437,   0.01497),
("H5",  "H",  0.3570,   0.5815,   0.0024 ),
("C6",  "C",  0.2800,   0.7362,   0.10954),
("H6",  "H",  0.3970,   0.7357,   0.1618 ),
("N1",  "N",  0.0528,   0.5529,  -0.15989),
("H2p", "H", -0.0730,   0.5440,  -0.1914 ),
("C7",  "C",  0.1792,   0.4882,  -0.21942),
("O2",  "O",  0.3532,   0.49635, -0.19141),
("C8",  "C",  0.0934,   0.4023,  -0.3234 ),
("H8A", "H", -0.0343,   0.4292,  -0.3436 ),
("H8B", "H",  0.1001,   0.3034,  -0.3055 ),
("H8C", "H",  0.1611,   0.4207,  -0.3858 ),
]

# Validate that .str atom names match the ASU list
asu_names = [n for (n,_,_,_,_) in asu]
missing = [n for n in atom_types.keys() if n not in asu_names]
if missing:
    sys.exit("ERROR: .str atom names not found in embedded APAP ASU: " + ", ".join(missing))

def sym_ops(x,y,z):
    # P21/n (Hall: -P 2yn), 4 symmetry ops (Z=4)
    return [
        ( x,               y,               z              ),
        (-x + 0.5,        y + 0.5,        -z + 0.5        ),
        (-x,              -y,              -z              ),
        ( x - 0.5,       -y - 0.5,         z - 0.5        ),
    ]

def frac_to_cart(fx, fy, fz):
    X = fx*ax + fy*bx + fz*cx
    Y = fx*ay + fy*by + fz*cy
    Z = fx*az + fy*bz + fz*cz
    return (X, Y, Z)

# Build one primitive cell: 4 molecules (each 20 atoms)
# We'll store a per-molecule local list of atoms [(name, elem, type, charge, x,y,z)]
mol_local = []   # list of 4 molecules; each is a list of per-atom dicts
for op_idx in range(4):
    atoms_m = []
    for (name, elem, fx, fy, fz) in asu:
        (u,v,w) = sym_ops(fx,fy,fz)[op_idx]
        x,y,z = frac_to_cart(u,v,w)
        atoms_m.append({
            "name":   name,
            "elem":   elem,
            "type":   atom_types[name],
            "charge": atom_chg[name],
            "x": x, "y": y, "z": z
        })
    mol_local.append(atoms_m)

# Template connectivity on one molecule (use .str bonds in the ASU name space)
name_to_idx = {n:i+1 for i,(n,_,_,_,_) in enumerate(asu)}  # local index 1..20
bonds_local = [(name_to_idx[a], name_to_idx[b]) for (a,b) in bonds_tpl]

# Helper: build adjacency for angles/dihedrals
def build_adj(natoms, bonds):
    adj = [[] for _ in range(natoms+1)]
    for i,j in bonds:
        adj[i].append(j)
        adj[j].append(i)
    return adj

def gen_angles(natoms, bonds):
    # angles are i-j-k with (i<k) to avoid duplicates; j is the vertex
    adj = build_adj(natoms, bonds)
    seen = set()
    out = []
    for j in range(1, natoms+1):
        nbrs = adj[j]
        for a in range(len(nbrs)):
            for b in range(a+1, len(nbrs)):
                i, k = nbrs[a], nbrs[b]
                key = tuple(sorted((i,k)) + [j])  # dedupe
                if key in seen: continue
                seen.add(key)
                out.append((i, j, k))
    return out

def gen_dihedrals(natoms, bonds):
    # dihedral is i-j-k-l for paths of length 3 with unique atoms
    adj = build_adj(natoms, bonds)
    out = []
    seen = set()
    for j in range(1, natoms+1):
        for i in adj[j]:
            if i == j: continue
            for k in adj[j]:
                if k == j or k == i: continue
                for l in adj[k]:
                    if l in (i,j,k): continue
                    key = (i,j,k,l)
                    # Deduplicate reverse (l,k,j,i)
                    key_min = min(key, key[::-1])
                    if key_min in seen: continue
                    # Ensure i-j and j-k and k-l are bonds
                    if (i,j) not in bonds and (j,i) not in bonds: continue
                    if (j,k) not in bonds and (k,j) not in bonds: continue
                    if (k,l) not in bonds and (l,k) not in bonds: continue
                    seen.add(key_min)
                    out.append(key)
    return out

angles_local   = gen_angles(20, bonds_local)
dihedrals_local= gen_dihedrals(20, bonds_local)
# impropers_local from .str names
impropers_local = []
for (a,b,c,d) in improp_tpl:
    impropers_local.append((
        name_to_idx[a], name_to_idx[b], name_to_idx[c], name_to_idx[d]
    ))

# ------------------ Type indexing ------------------
# Atom types: numeric ID per unique CGenFF atom type
unique_atom_types = sorted({atom_types[n] for n in atom_types})
atype_id = {t:i+1 for i,t in enumerate(unique_atom_types)}

# For bonds/angles/dihedrals/impropers we will create numeric types
# by their (atom-type) pattern. The coefficients are NOT provided here.
def bond_key(i,j, atoms_this_mol):
    t1 = atoms_this_mol[i-1]["type"]; t2 = atoms_this_mol[j-1]["type"]
    return tuple(sorted((t1,t2)))

def angle_key(i,j,k, atoms_this_mol):
    t1 = atoms_this_mol[i-1]["type"]; t2 = atoms_this_mol[j-1]["type"]; t3 = atoms_this_mol[k-1]["type"]
    return (t1,t2,t3)

def dihed_key(i,j,k,l, atoms_this_mol):
    t1 = atoms_this_mol[i-1]["type"]; t2 = atoms_this_mol[j-1]["type"]
    t3 = atoms_this_mol[k-1]["type"]; t4 = atoms_this_mol[l-1]["type"]
    # keep direction (proper dihedral), but canonicalize symmetric reverse pairing later in coeff assignment if desired
    return (t1,t2,t3,t4)

def imprp_key(i,j,k,l, atoms_this_mol):
    t1 = atoms_this_mol[i-1]["type"]; t2 = atoms_this_mol[j-1]["type"]
    t3 = atoms_this_mol[k-1]["type"]; t4 = atoms_this_mol[l-1]["type"]
    return (t1,t2,t3,t4)

bond_keys_set   = set()
angle_keys_set  = set()
dihed_keys_set  = set()
imprp_keys_set  = set()

# Collect type keys from ONE molecule (type sets same for all molecules)
for (i,j) in bonds_local:
    bond_keys_set.add(bond_key(i,j, mol_local[0]))
for (i,j,k) in angles_local:
    angle_keys_set.add(angle_key(i,j,k, mol_local[0]))
for (i,j,k,l) in dihedrals_local:
    dihed_keys_set.add(dihed_key(i,j,k,l, mol_local[0]))
for (i,j,k,l) in impropers_local:
    imprp_keys_set.add(imprp_key(i,j,k,l, mol_local[0]))

bondtype_id  = {key:i+1 for i,key in enumerate(sorted(bond_keys_set))}
angletype_id = {key:i+1 for i,key in enumerate(sorted(angle_keys_set))}
dihedtype_id = {key:i+1 for i,key in enumerate(sorted(dihed_keys_set))}
imprptype_id = {key:i+1 for i,key in enumerate(sorted(imprp_keys_set))}

# ------------------ Replicate to supercell ------------------
atoms_out    = []  # tuples: (id, mol, atype_num, charge, x,y,z)
bonds_out    = []  # (id, btype_num, i, j)
angles_out   = []  # (id, atype_num, i, j, k)
diheds_out   = []  # (id, dtype_num, i, j, k, l)
improps_out  = []  # (id, itype_num, i, j, k, l)

nat_per_mol  = 20
nmol_per_cell= 4
nat_per_cell = nat_per_mol * nmol_per_cell

aid_counter = 1
bid_counter = 1
ang_counter = 1
dih_counter = 1
imp_counter = 1

# mass by element (used for "Masses" per atom type)
elem_mass = {"H":1.008, "C":12.011, "N":14.007, "O":15.999}

def elem_of_atom(atom_record):
    return atom_record["elem"]

# Box
xlo=ylo=zlo=0.0
Lx = nx * a
Ly = ny * b
Lz = nz * cz
xy = 0.0
xz = nz * cx
yz = 0.0

for kz in range(nz):
    for ky in range(ny):
        for kx in range(nx):
            # tile offset in Cartesian using lattice vectors
            dx = kx*ax + ky*bx + kz*cx
            dy = kx*ay + ky*by + kz*cy
            dz = kx*az + ky*bz + kz*cz

            tile_index = kz*(ny*nx) + ky*nx + kx
            molid_offset = tile_index*nmol_per_cell

            for m in range(nmol_per_cell):
                atoms_m = mol_local[m]
                mol_id = molid_offset + (m+1)
                atom_offset = tile_index*nat_per_cell + m*nat_per_mol

                # Atoms
                for i in range(nat_per_mol):
                    rec = atoms_m[i]
                    atype_num = atype_id[rec["type"]]
                    q = rec["charge"]
                    x = rec["x"] + dx; y = rec["y"] + dy; z = rec["z"] + dz
                    atoms_out.append( (aid_counter, mol_id, atype_num, q, x, y, z) )
                    aid_counter += 1

                # Bonds
                for (i,j) in bonds_local:
                    key = bond_key(i,j, atoms_m)
                    btype_num = bondtype_id[key]
                    ai = atom_offset + i
                    aj = atom_offset + j
                    bonds_out.append( (bid_counter, btype_num, ai, aj) )
                    bid_counter += 1

                # Angles
                for (i,j,k) in angles_local:
                    key = angle_key(i,j,k, atoms_m)
                    atnum = angletype_id[key]
                    ai = atom_offset + i
                    aj = atom_offset + j
                    ak = atom_offset + k
                    angles_out.append( (ang_counter, atnum, ai, aj, ak) )
                    ang_counter += 1

                # Dihedrals
                for (i,j,k,l) in dihedrals_local:
                    key = dihed_key(i,j,k,l, atoms_m)
                    dtnum = dihedtype_id[key]
                    ai = atom_offset + i
                    aj = atom_offset + j
                    ak = atom_offset + k
                    al = atom_offset + l
                    diheds_out.append( (dih_counter, dtnum, ai, aj, ak, al) )
                    dih_counter += 1

                # Impropers (from .str)
                for (i,j,k,l) in impropers_local:
                    key = imprp_key(i,j,k,l, atoms_m)
                    itnum = imprptype_id[key]
                    ai = atom_offset + i
                    aj = atom_offset + j
                    ak = atom_offset + k
                    al = atom_offset + l
                    improps_out.append( (imp_counter, itnum, ai, aj, ak, al) )
                    imp_counter += 1

natoms    = len(atoms_out)
nbonds    = len(bonds_out)
nangles   = len(angles_out)
ndihed    = len(diheds_out)
nimprop   = len(improps_out)

# ------------------ Write LAMMPS data ------------------
with open(out_data, "w") as w:
    w.write("# LAMMPS data file for Acetaminophen Form I (CGenFF types/charges)\n")
    w.write("# Source .str: {}\n".format(in_str))
    w.write("# Supercell: nx={} ny={} nz={}\n".format(nx,ny,nz))
    w.write("# Pair/Bond/Angle/Dihedral/Improper COEFFS NOT INCLUDED. Use CHARMM/CGenFF params in input.\n")
    w.write("#\n# Atom type map (numeric -> CGenFF type):\n")
    for tstr, tid in sorted(atype_id.items(), key=lambda kv: kv[1]):
        w.write("#   {:>3d} : {}\n".format(tid, tstr))
    w.write("#\n# Bond type keys (numeric -> (t1,t2) sorted):\n")
    for key, bid in sorted(bondtype_id.items(), key=lambda kv: kv[1]):
        w.write("#   {:>3d} : {} - {}\n".format(bid, key[0], key[1]))
    w.write("#\n# Angle type keys (numeric -> (t1,t2,t3)):\n")
    for key, aid_ in sorted(angletype_id.items(), key=lambda kv: kv[1]):
        w.write("#   {:>3d} : {}-{}-{}\n".format(aid_, key[0], key[1], key[2]))
    w.write("#\n# Dihedral type keys (numeric -> (t1,t2,t3,t4)):\n")
    for key, did in sorted(dihedtype_id.items(), key=lambda kv: kv[1]):
        w.write("#   {:>3d} : {}-{}-{}-{}\n".format(did, key[0], key[1], key[2], key[3]))
    w.write("#\n# Improper type keys (numeric -> (t1,t2,t3,t4)):\n")
    for key, iid in sorted(imprptype_id.items(), key=lambda kv: kv[1]):
        w.write("#   {:>3d} : {}-{}-{}-{}\n".format(iid, key[0], key[1], key[2], key[3]))
    w.write("\n")

    # Counts
    w.write("{} atoms\n".format(natoms))
    if nbonds:  w.write("{} bonds\n".format(nbonds))
    if nangles: w.write("{} angles\n".format(nangles))
    if ndihed:  w.write("{} dihedrals\n".format(ndihed))
    if nimprop: w.write("{} impropers\n".format(nimprop))
    w.write("\n")

    # Type counts
    w.write("{} atom types\n".format(len(atype_id)))
    if nbonds:  w.write("{} bond types\n".format(len(bondtype_id)))
    if nangles: w.write("{} angle types\n".format(len(angletype_id)))
    if ndihed:  w.write("{} dihedral types\n".format(len(dihedtype_id)))
    if nimprop: w.write("{} improper types\n".format(len(imprptype_id)))
    w.write("\n")

    # ------------------ Box (floats) ------------------
    Lx = float(nx) * float(a_lat)
    Ly = float(ny) * float(b_lat)
    Lz = float(nz) * float(c_lat * math.sin(beta))
    xz = float(nz) * float(c_lat * math.cos(beta))
    xy = 0.0
    yz = 0.0

    # ------------------ Write triclinic box ------------------
    w.write("{:.8f} {:.8f} xlo xhi\n".format(0.0, float(Lx)))
    w.write("{:.8f} {:.8f} ylo yhi\n".format(0.0, float(Ly)))
    w.write("{:.8f} {:.8f} zlo zhi\n".format(0.0, float(Lz)))
    w.write("{:.8f} {:.8f} {:.8f} xy xz yz\n".format(float(xy), float(xz), float(yz)))
    w.write("\n")


    # Masses (per atom type; use element mass of any example atom of that type)
    # Build type->element by inspecting one appearance in mol_local[0]
    t_to_elem = {}
    for rec in mol_local[0]:
        t = rec["type"]; el = rec["elem"]
        t_to_elem.setdefault(t, el)
    w.write("Masses\n\n")
    for tstr, tid in sorted(atype_id.items(), key=lambda kv: kv[1]):
        el = t_to_elem.get(tstr, "C")
        mass = elem_mass.get(el, 12.011)
        w.write("{} {:.4f}    # {}\n".format(tid, mass, tstr))
    w.write("\n")

    # Atoms (atom-ID mol-ID atom-type charge x y z)
    w.write("Atoms  # atom-ID mol-ID atom-type charge x y z\n\n")
    for rec in atoms_out:
        aid, mol, atid, q, x, y, z = rec
        w.write("{} {} {} {:.6f} {:.8f} {:.8f} {:.8f}\n".format(aid, mol, atid, q, x, y, z))
    w.write("\n")

    # Bonds
    if nbonds:
        w.write("Bonds  # id type i j\n\n")
        for (bid, btype, i, j) in bonds_out:
            w.write("{} {} {} {}\n".format(bid, btype, i, j))
        w.write("\n")

    # Angles
    if nangles:
        w.write("Angles  # id type i j k\n\n")
        for (aid_, atype, i, j, k) in angles_out:
            w.write("{} {} {} {} {}\n".format(aid_, atype, i, j, k))
        w.write("\n")

    # Dihedrals
    if ndihed:
        w.write("Dihedrals  # id type i j k l\n\n")
        for (did, dtype, i, j, k, l) in diheds_out:
            w.write("{} {} {} {} {} {}\n".format(did, dtype, i, j, k, l))
        w.write("\n")

    # Impropers
    if nimprop:
        w.write("Impropers  # id type i j k l\n\n")
        for (iid, itype, i, j, k, l) in improps_out:
            w.write("{} {} {} {} {} {}\n".format(iid, itype, i, j, k, l))
        w.write("\n")

print(f"Wrote: {out_data}")
print(f"Atoms={natoms} Bonds={nbonds} Angles={nangles} Dihedrals={ndihed} Impropers={nimprop}")
print("REMINDER: No coefficients in data file. In LAMMPS, load CHARMM/CGenFF params and assign coeffs to the numeric types listed in the header comments.")
