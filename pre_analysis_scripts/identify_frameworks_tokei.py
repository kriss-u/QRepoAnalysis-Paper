import configparser
import csv
import fnmatch
import json
import os
import subprocess
import re
import tomli

# Input
REPOS_DIR = "../missed-clones"

# Output
EMPTY_FILE = "empty_missed.csv"
MAPPED_FILE = "mapped_missed.csv"
UNMAPPED_FILE = "unmapped_missed.csv"

MAPPED_STAGE2_FILE = "mapped_missed_stage2.csv"
UNMAPPED_STAGE2_FILE = "unmapped_missed_stage2.csv"
EMPTY_STAGE2_FILE = "empty_missed_stage2.csv"

# Framework information extracted from project knowledge
FRAMEWORK_INFO = {
    "python": [
        ("Alibaba Cloud Quantum Development Platform", "acqdp"),
        ("Aero Attention", "aeroattention"),
        ("Braket", "amazon-braket-sdk"),
        ("Azure MGMT Quantum", "azure-mgmt-quantum"),
        ("Azure Core", "azure-core"),
        ("Azure Quantum", "azure-quantum"),
        ("Azure Quantum", "azure.quantum.qiskit"),
        ("Blueqat", "blueqat"),
        ("BQSkit", "bqskit"),
        ("Braandket", "braandket"),
        ("Cirq", "cirq"),
        ("Cirq Core", "cirq-core"),
        ("Cuda Quantum", "cuda-quantum"),
        ("CuQuantum", "cuquantum"),
        ("DigitalSoul", "DigitalSoul"),
        ("Ocean", "dwave-ocean-sdk"),
        ("DWave System", "dwave-system"),
        ("GraphEPP", "graphepp"),
        ("Graphix", "graphix"),
        ("Guppy Lang", "guppylang"),
        ("IBMQuantumExperience", "IBMQuantumExperience"),
        ("IQM Client", "iqm-client"),
        ("Mind Quantum", "mindquantum"),
        ("MQT Debugger", "mqt-debugger"),
        ("MQT QCEC", "mqt-qcec"),
        ("MQT Qudits", "mqt.qudits"),
        ("MQT MiSiM", "mqt.misim"),
        ("NumQi", "numqi"),
        ("OpenFermion", "openfermion"),
        ("OpenQASM", "openqasm3"),
        ("Openqemist", "openqemist"),
        ("OpenQL", "openql"),
        ("OQpy", "oqpy"),
        ("Orquestra Quantum", "orquestra-quantum"),
        ("PauliOpt", "pauliopt"),
        ("PennyLane", "pennylane"),
        ("PennyLange-QuantumInspire Plugin", "pennylane-quantuminspire2"),
        ("ProjectQ", "projectq"),
        ("PyQCS", "pyqcs"),
        ("PyQIR", "pyqir"),
        ("PyQrack", "pyqrack"),
        ("PyQTorch", "pyqtorch"),
        ("PyQudit", "pyqudit"),
        ("PyQuil", "pyquil"),
        ("PyStaq", "pystaq"),
        ("pytket", "pytket"),
        ("Pyzx", "pyzx"),
        ("QAOA Framework", "QAOA"),
        ("QArray", "qarray"),
        ("QArray Rust Core", "qarray-rust-core"),
        ("QAT Lang", "qat-lang"),
        ("QCGPU", "qcgpu"),
        ("Qbraid Core", "qbraid-core"),
        ("Qbraid QIR", "qbraid-qir"),
        ("QCircPy", "qcircpy"),
        ("QCoDeS", "qcodes"),
        ("QCLight", "qclight"),
        ("QCommunity", "qcommunity"),
        ("Quantum Cloud Services", "qcs-sdk-python"),
        ("Quantum Computing Research Utils", "qcutils"),
        ("Qib", "qib"),
        ("Qibo", "qibo"),
        ("QICK", "qick"),
        ("QiController", "qiclib"),
        ("Numpy Quantum Info", "qinfo"),
        ("Qiskit", "qiskit"),
        ("Qiskit Aer", "qiskit-aer"),
        ("Qiskit Aqua", "qiskit-aqua"),
        ("Qiskit IBMQ Provider", "qiskit-ibmq-provider"),
        ("Qiskit Ignis", "qiskit-ignis"),
        ("Qiskit Metal", "qiskit-metal"),
        ("Qiskit Nature", "qiskit-nature"),
        ("Qiskit Terra", "qiskit-terra"),
        ("Qiskit Flow", "qiskitflow"),
        ("Qkit", "qkit"),
        ("Qlasskit", "qlasskit"),
        ("QM_QUA", "qm_qua"),
        ("QM-QUA", "qm-qua"),
        ("QoMo", "qomo"),
        ("QPtomographer", "QPtomographer"),
        ("Qradient", "qradient"),
        ("Quantum Random Number Generator", "qrng"),
        ("Qsharp", "qsharp"),
        ("Qsteed", "qsteed"),
        ("QSystem", "qsystem"),
        ("Qualang Tools", "qualang-tools"),
        ("Quantify Core", "quantify-core"),
        ("Quantumcat", "quantumcat"),
        ("QuantumCatch", "quantumcatch"),
        ("Qubit Mapping", "qubitmapping"),
        ("Quantum Tomography", "Quantum-Tomography"),
        ("QuCircuit", "qucircuit"),
        ("QuDiet", "QuDiet"),
        ("Quil", "quil"),
        ("Quimb", "quimb"),
        ("Qulacs", "qulacs"),
        ("QuPulse", "qupulse"),
        ("QuPy", "qupy"),
        ("Quri Parts", "quri-parts"),
        ("Qusimulator", "qusimulator"),
        ("Qusource", "qusource"),
        ("QuTiP", "qutip"),
        ("Qutritium", "qutritium"),
        ("ReQuSim", "requsim"),
        ("Rqcopt", "rqcopt"),
        ("SFQLib", "sfqlib"),
        ("Skqulacs", "skqulacs"),
        ("SpinQKit", "spinqkit"),
        ("Squanch", "SQUANCH"),
        ("Stacasso", "stacasso"),
        ("Strawberry Fields", "strawberryfields"),
        ("Tensorflow Quantum", "tensorflow-quantum"),
        ("Torch Quantum", "torchquantum"),
        ("Toqito", "toqito"),
        ("TQsim", "tqsim"),
        ("Z Quantum Core", "z-quantum-core"),
        ("Z Quantum Core Framework", "zquantum"),
        ("QuantestPy", "quantestpy"),
        ("Zyglrox", "zyglrox"),
        ("Quantastica Qconvert", "quantastica-qconvert"),
        ("QuCAT", "qucat"),
        ("QutiePy", "QutiePy"),
        ("PyQASM", "pyqasm"),
        ("Tequila", "tequila-basic"),
        ("QuJAX", "qujax"),
        ("QM Octave", "qm-octave"),
        ("QGates", "qgates"),
        ("Qpic", "qpic"),
        ("DC  Qiskit Algorithms", "dc-qiskit-algorithms"),
        ("Qisjob", "qisjob"),
        ("QcPy", "qcpython"),
        ("Pyvoqc", "pyvoqc"),
        ("QASM Testsuite", "qasm-testsuite"),
        ("YAQQ", "yaqq"),
        ("Quantum Xir", "quantum-xir"),
        ("Qailo", "qailo"),
        ("Quanthon", "Quanthon"),
        ("Quantify Scheduler", "quantify-scheduler"),
        ("QCompute", "qcompute"),
        ("Myqlm Fermion", "myqlm-fermion"),
        ("QOPT", "qopt"),
        ("qMuVi", "qMuVi"),
        ("Doki Mowstyl", "doki-Mowstyl"),
        ("QIP", "qip"),
        ("LibQASM", "libqasm"),
        ("OpenSquirrel", "opensquirrel"),
        ("Logiq", "logiq"),
        ("QuIS", "quis"),
        ("QANDLE", "qandle"),
        ("Quanlse", "Quanlse"),
        ("Horqrux", "horqrux"),
        ("Qonduit", "qonduit"),
        ("Quantum Grove", "quantum-grove"),
        ("Forest Openfermion", "forestopenfermion"),
        ("qLDPC", "qLDPC"),
        ("Qvantum", "qvantum"),
        ("Qudotpy", "qudotpy"),
        ("QPanda", "pyqpanda"),
    ],
    "c++": [
        ("CUDA-Q", "cuda-quantum"),
        ("Forest SDK", "forest-sdk"),
        ("HyQuas", "QCSimulator"),
        ("QPlayer", "qplayer"),
        ("QPtomographer", "QPtomographer"),
        ("Quantum++", "qpp"),
        ("Qrack", "qrack"),
        ("QuEST", "quest"),
        ("Quilc", "quilc"),
        ("QuISP", "quisp"),
        ("XACC", "xacc"),
        ("DQCsim cQASM", "dqcsim-cqasm"),
        ("MQT QMap", "mqt-qmap"),
        ("MQT qusat", "qusat"),
        ("Stim", "stim"),
        ("MQT Core", "mqt-core"),
        ("MQT DDSIM", "mqt-ddsim"),
        ("QPANDA", "QPANDA"),
    ],
    "c": [
        ("Forest SDK", "forest-sdk"),
        ("Qrack", "qrack"),
        ("Quilc", "quilc"),
        ("QuISP", "quisp"),
    ],
    "q#": [("Q#", "Microsoft.Quantum")],
    "silq": [("Silq", "silq")],
    "julia": [
        ("Braket", "Braket"),
        ("PastaQ", "PastaQ"),
        ("QuantumInformation", "QuantumInformation"),
        ("QuantumOptics", "QuantumOptics"),
        ("VQC", "VQC"),
        ("Yao.jl", "Yao"),
    ],
    "rust": [
        ("OpenQASM", "openqasm"),
        ("Classical to Quantum Computing Compiler", "quantpiler"),
        ("Quil", "qcs"),
        ("QEC", "qecp"),
        ("QoQo", "qoqo"),
        ("Quil Rust", "quil-rs"),
        ("Quizx", "quizx"),
        ("Rust Only QoQO", "roqoqo"),
        ("RustQIP", "rustqip"),
        ("Rasqal", "rasqal"),
        ("Tket JSON Rust", "tket-json-rs"),
        ("Qsharp", "qsharp"),
        ("Azure SDK For Rust", "azure_sdk_for_rust"),
    ],
    "lisp": [
        ("Forest SDK", "forest-sdk"),
        ("QCL", "qcl"),
        ("Quilc", "quilc"),
    ],
}

PACKAGE_FILES = {
    "python": [
        "Pipfile",
        "*requirements*.txt",
        "requires*.txt",
        "required*.txt",
        "setup.py",
        "pyproject.toml",
        "setup.cfg",
    ],
    "c++": ["CMakeLists.txt", "Makefile", "conanfile.txt"],
    "c": ["CMakeLists.txt", "Makefile"],
    "lisp": ["Makefile"],
    "julia": ["Project.toml"],
    "rust": ["Cargo.toml"],
}


# Programming languages identified from linguist's languages.yml
PROGRAMMING_LANGUAGES = {
    "python",
    "c++",
    "c",
    "julia",
    "rust",
    "q#",
    "silq",
    "c#",
    "f#",
    "go",
    "java",
    "javascript",
    "typescript",
    "scala",
    "ruby",
    "haskell",
    "ocaml",
    "swift",
    "fortran",
    "kotlin",
    "matlab",
    "r",
    "php",
    "perl",
    "lua",
    "erlang",
    "elixir",
    "elm",
    "nim",
    "d",
    "dart",
    "crystal",
    "f*",
    "idris",
    "mercury",
    "pureScript",
}

# Quantum-specific languages identified from languages.yml
QUANTUM_LANGUAGES = {
    "openqasm",
    "q#",
    "qsharp",  # Microsoft's Q#
    "silq",  # ETH Zurich's Silq
    "quil",  # Rigetti's Quil
    "quilc",  # Quil compiler
    "quipper",  # Haskell-based quantum language
}


def parse_setup_cfg(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)

    name = None
    if "metadata" in config and "name" in config["metadata"]:
        name = config["metadata"]["name"].strip()

    requirements = []
    if "options" in config and "install_requires" in config["options"]:
        install_requires = config["options"]["install_requires"]
        requirements = [
            line.strip() for line in install_requires.split("\n") if line.strip()
        ]

    return name, requirements


def run_tokei(repo_path):
    result = subprocess.run(
        [
            "tokei",
            "-o",
            "json",
            "-t=Rust,C++,C,C Header,C++ Header,Python,JavaScript,TypeScript,Java,Kotlin,Scala,Go,Ruby,PHP,Swift,Objective-C,C#,F#,Visual Basic,R,Julia,Haskell,OCaml,Erlang,Elixir,Clojure,Lisp,Scheme,Lua,Perl,Dart,Groovy,Fortran,COBOL,Ada,Prolog,D,Nim,Crystal,Zig,Odin,Vala,Haxe,Racket,Elm,PureScript,Idris,Agda,Coq,Q#,Silq,Qcl",
            repo_path,
        ],
        capture_output=True,
        text=True,
    )

    try:
        data = json.loads(result.stdout)
        if not data or (len(data) == 1 and "Total" in data):
            print(f"No language-specific data found in tokei output for {repo_path}")
            return None

        data.pop("Total", None)

        if not data:
            print(
                f"No language-specific data found after removing Total for {repo_path}"
            )
            return None

        max_code_language = max(
            data.items(),
            key=lambda x: x[1].get("code", 0) if isinstance(x[1], dict) else 0,
        )

        language_name = max_code_language[0].lower()

        if language_name == "c header":
            language_name = "c"

        elif language_name == "c++ header":
            language_name = "c++"

        language_stats = max_code_language[1]

        return (
            language_name,
            len(language_stats.get("reports", [])),
            language_stats.get("code", 0),
            language_stats.get("comments", 0),
            language_stats.get("blanks", 0),
        )
    except json.JSONDecodeError:
        print(f"Error parsing tokei output for {repo_path}")
        return None
    except Exception as e:
        print(f"Unexpected error processing tokei output for {repo_path}: {str(e)}")
        return None


def find_framework_files(repo_path, language, max_depth=4):
    package_files = PACKAGE_FILES.get(language, [])
    frameworks = FRAMEWORK_INFO.get(language, [])

    def search_dir(current_path, current_depth):
        if current_depth > max_depth:
            return None

        try:
            # First check all matching files in current directory
            for file in os.listdir(current_path):
                file_path = os.path.join(current_path, file)
                if not os.path.isfile(file_path):
                    continue

                # Check if file matches any of our glob patterns
                if any(fnmatch.fnmatch(file, pattern) for pattern in package_files):
                    matched_frameworks = check_file_for_frameworks(
                        file_path, frameworks, language
                    )
                    if matched_frameworks:
                        return (file_path, matched_frameworks)

            # If no frameworks found in current directory, search subdirectories
            for item in os.listdir(current_path):
                item_path = os.path.join(current_path, item)
                if os.path.isdir(item_path):
                    result = search_dir(item_path, current_depth + 1)
                    if result:
                        return result

        except Exception as e:
            print(f"Error searching directory {current_path}: {str(e)}")

        return None

    result = search_dir(repo_path, 1)
    return [(result[0], result[1])] if result else []


def check_file_for_frameworks(file_path, frameworks, language):
    matched_frameworks = []
    try:
        file_name = os.path.basename(file_path)

        # Handle TOML files (pyproject.toml, Pipfile)
        if file_name.lower() in ["pyproject.toml", "pipfile", "cargo.toml"]:
            try:
                with open(file_path, "rb") as f:
                    try:
                        content = tomli.load(f)
                    except tomli.TOMLDecodeError as e:
                        print(f"TOML parsing error in {file_path}: {str(e)}")
                        return []

                    if file_name.lower() == "cargo.toml":
                        # Process Rust dependencies
                        def check_deps(deps):
                            if not deps or not isinstance(deps, dict):
                                return

                            for package_name in deps.keys():
                                package_name = package_name.lower()
                                for framework, fw_package_name in frameworks:
                                    if fw_package_name.lower() == package_name:
                                        matched_frameworks.append(
                                            (framework, fw_package_name)
                                        )

                        dependencies = content.get("dependencies", {})
                        check_deps(dependencies)

                        dev_dependencies = content.get("dev-dependencies", {})
                        check_deps(dev_dependencies)

                        build_dependencies = content.get("build-dependencies", {})
                        check_deps(build_dependencies)

                    elif file_name.lower() == "pipfile":
                        packages = content.get("packages", {})
                        dev_packages = content.get("dev-packages", {})

                        def extract_package_names(packages_dict):
                            for package_name, spec in packages_dict.items():
                                package_name = package_name.lower()
                                for framework, fw_package_name in frameworks:
                                    if fw_package_name.lower() == package_name:
                                        matched_frameworks.append(
                                            (framework, fw_package_name)
                                        )

                        extract_package_names(packages)
                        extract_package_names(dev_packages)

                    else:  # pyproject.toml handling
                        project_name = (
                            content.get("project", {}).get("name", "").lower()
                            or content.get("tool", {})
                            .get("poetry", {})
                            .get("name", "")
                            .lower()
                        )

                        if project_name:
                            for framework, package_name in frameworks:
                                if package_name.lower() == project_name:
                                    matched_frameworks.append((framework, package_name))

                        def safe_check_deps(deps):
                            if not deps:
                                return

                            if isinstance(deps, list):
                                for dep in deps:
                                    try:
                                        package = (
                                            dep.split()[0].lower()
                                            if isinstance(dep, str)
                                            else ""
                                        )
                                        for framework, package_name in frameworks:
                                            if package_name.lower() == package:
                                                matched_frameworks.append(
                                                    (framework, package_name)
                                                )
                                    except (IndexError, AttributeError):
                                        continue

                            elif isinstance(deps, dict):
                                for package in deps.keys():
                                    for framework, package_name in frameworks:
                                        if package_name.lower() == package.lower():
                                            matched_frameworks.append(
                                                (framework, package_name)
                                            )

                        safe_check_deps(
                            content.get("project", {}).get("dependencies", [])
                        )
                        optional_deps = content.get("project", {}).get(
                            "optional-dependencies", {}
                        )
                        for deps in optional_deps.values():
                            safe_check_deps(deps)

                        poetry_section = content.get("tool", {}).get("poetry", {})
                        if poetry_section:
                            safe_check_deps(poetry_section.get("dependencies", {}))

                            groups = poetry_section.get("group", {})
                            for group in groups.values():
                                if isinstance(group, dict):
                                    safe_check_deps(group.get("dependencies", {}))

                        safe_check_deps(
                            content.get("build-system", {}).get("requires", [])
                        )

            except Exception as e:
                print(f"Error processing TOML file {file_path}: {str(e)}")
                return []

        elif language == "python":
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()

                if (
                    fnmatch.fnmatch(file_name.lower(), "*requirements*.txt")
                    or fnmatch.fnmatch(file_name.lower(), "requires*.txt")
                    or fnmatch.fnmatch(file_name.lower(), "required*.txt")
                ):
                    content_lower = content.lower()
                    for framework, package_name in frameworks:
                        if re.search(
                            rf"^{re.escape(package_name.lower())}(?:[=<>]|$)",
                            content_lower,
                            re.MULTILINE | re.IGNORECASE,
                        ):
                            matched_frameworks.append((framework, package_name))

                elif file_name.lower() == "setup.py":
                    content_lower = content.lower()
                    for framework, package_name in frameworks:
                        if re.search(
                            rf"['\"]({re.escape(package_name.lower())})(?:[><=].+)?['\"]",
                            content_lower,
                            re.IGNORECASE,
                        ):
                            matched_frameworks.append((framework, package_name))

                elif file_name.lower() == "setup.cfg":
                    try:
                        name, requirements = parse_setup_cfg(file_path)
                        if name:
                            for framework, package_name in frameworks:
                                if package_name.lower() == name.lower():
                                    matched_frameworks.append((framework, package_name))

                        for requirement in requirements:
                            for framework, package_name in frameworks:
                                if package_name.lower() in requirement.lower():
                                    matched_frameworks.append((framework, package_name))
                    except Exception as e:
                        print(f"Error processing setup.cfg file {file_path}: {str(e)}")

            except Exception as e:
                print(f"Error processing Python file {file_path}: {str(e)}")
                return matched_frameworks

        elif language in ["c", "c++", "lisp"]:
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read().lower()
                    for framework, package_name in frameworks:
                        if package_name.lower() in content:
                            matched_frameworks.append((framework, package_name))
            except Exception as e:
                print(f"Error processing C/C++/Lisp file {file_path}: {str(e)}")
                return matched_frameworks

        elif language == "julia":
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read().lower()
                    for framework, package_name in frameworks:
                        if package_name.lower() in content:
                            matched_frameworks.append((framework, package_name))
            except Exception as e:
                print(f"Error processing Julia file {file_path}: {str(e)}")
                return matched_frameworks

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []

    return matched_frameworks


def search_frameworks(repo_path, language):
    framework_files = find_framework_files(repo_path, language)

    if not framework_files:
        return []

    # Return frameworks from the first file that had matches
    _, matched_frameworks = framework_files[0]

    seen = set()
    return [x for x in matched_frameworks if not (x in seen or seen.add(x))]


def main(repos_dir):
    with open(MAPPED_FILE, "w", newline="") as mapped_file, open(
        UNMAPPED_FILE, "w", newline=""
    ) as unmapped_file, open(EMPTY_FILE, "w", newline="") as empty_file:

        mapped_writer = csv.writer(mapped_file)
        unmapped_writer = csv.writer(unmapped_file)
        empty_writer = csv.writer(empty_file)

        mapped_writer.writerow(
            ["repo_name", "language", "packages", "files", "code", "comments", "blanks"]
        )
        unmapped_writer.writerow(["repo_name", "language"])
        empty_writer.writerow(["repo_name"])

        for repo in os.listdir(repos_dir):
            repo_path = os.path.join(repos_dir, repo)
            if not os.path.isdir(repo_path):
                continue

            tokei_result = run_tokei(repo_path)
            if tokei_result is None:
                empty_writer.writerow([repo])
                continue

            language, files, code, comments, blanks = tokei_result

            frameworks = (
                search_frameworks(repo_path, language)
                if language in FRAMEWORK_INFO
                else []
            )

            if frameworks:
                package_patterns = [framework[1] for framework in frameworks]
                mapped_writer.writerow(
                    [
                        repo,
                        language,
                        ";".join(package_patterns),
                        files,
                        code,
                        comments,
                        blanks,
                    ]
                )
            else:
                unmapped_writer.writerow([repo, language])

    print(
        f"Processing complete. Check {MAPPED_FILE}, {UNMAPPED_FILE}, and {EMPTY_FILE} for results."
    )


def run_linguist(repo_path):
    try:
        result = subprocess.run(
            ["github-linguist", repo_path], capture_output=True, text=True, check=True
        )

        for line in result.stdout.splitlines():
            if not line.strip():
                continue

            # Split line on whitespace and get language and code count
            # Format: "61.86%  282635     Python"
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            code_count = int(parts[1])  # Second column is the code count
            language = parts[-1].lower()  # Language is the last part

            if language in QUANTUM_LANGUAGES:
                return language, code_count

            if language in PROGRAMMING_LANGUAGES:
                return language, code_count

        return None

    except subprocess.CalledProcessError as e:
        print(f"Error running GitHub Linguist on {repo_path}: {str(e)}")
        return None


def process_unmapped_repos():
    unmapped_repos = []
    empty_repos = []

    try:
        with open(UNMAPPED_FILE, "r", newline="") as unmapped_file:
            reader = csv.reader(unmapped_file)
            next(reader)
            unmapped_repos.extend(row for row in reader)
    except Exception as e:
        print(f"Warning: Could not read {UNMAPPED_FILE}: {str(e)}")

    try:
        with open(EMPTY_FILE, "r", newline="") as empty_file:
            reader = csv.reader(empty_file)
            next(reader)
            empty_repos.extend(row for row in reader)
    except Exception as e:
        print(f"Warning: Could not read {EMPTY_FILE}: {str(e)}")

    try:
        with open(MAPPED_STAGE2_FILE, "w", newline="") as mapped_file, open(
            UNMAPPED_STAGE2_FILE, "w", newline=""
        ) as unmapped_stage2_file, open(
            EMPTY_STAGE2_FILE, "w", newline=""
        ) as empty_stage2_file:

            mapped_writer = csv.writer(mapped_file)
            unmapped_writer = csv.writer(unmapped_stage2_file)
            empty_writer = csv.writer(empty_stage2_file)

            mapped_writer.writerow(
                [
                    "repo_name",
                    "language",
                    "packages",
                    "files",
                    "code",
                    "comments",
                    "blanks",
                ]
            )
            unmapped_writer.writerow(["repo_name", "language"])
            empty_writer.writerow(["repo_name"])

            for repos in [unmapped_repos, empty_repos]:
                for row in repos:
                    try:
                        repo_name = row[0]
                        repo_path = os.path.join(REPOS_DIR, repo_name)

                        if not os.path.exists(repo_path):
                            print(
                                f"Warning: Repository path does not exist: {repo_path}"
                            )
                            continue

                        linguist_result = run_linguist(repo_path)

                        if linguist_result is None:
                            empty_writer.writerow([repo_name])
                            continue

                        language, code_count = linguist_result

                        # If it's a quantum language, treat it as its own framework
                        if language in QUANTUM_LANGUAGES:
                            mapped_writer.writerow(
                                [
                                    repo_name,
                                    language,
                                    language,  # Use language itself as package
                                    0,  # files count not available from linguist
                                    code_count,  # Use code count from linguist
                                    0,  # comments not available
                                    0,  # blanks not available
                                ]
                            )
                            continue

                        # For regular programming languages, check for frameworks
                        try:
                            frameworks = search_frameworks(repo_path, language)
                        except Exception as e:
                            print(
                                f"Warning: Error searching frameworks for {repo_name}: {str(e)}"
                            )
                            frameworks = []

                        if frameworks:
                            package_patterns = [
                                framework[1] for framework in frameworks
                            ]
                            mapped_writer.writerow(
                                [
                                    repo_name,
                                    language,
                                    ";".join(package_patterns),
                                    0,  # files not available
                                    code_count,  # Use code count from linguist
                                    0,  # comments not available
                                    0,  # blanks not available
                                ]
                            )
                        else:
                            unmapped_writer.writerow([repo_name, language])

                    except Exception as e:
                        print(
                            f"Warning: Error processing repository {row[0] if row else 'unknown'}: {str(e)}"
                        )
                        continue

    except Exception as e:
        print(f"Error opening output files: {str(e)}")

    print(
        f"Processing complete. Check {MAPPED_STAGE2_FILE}, {UNMAPPED_STAGE2_FILE}, and {EMPTY_STAGE2_FILE} for results."
    )


if __name__ == "__main__":
    main(REPOS_DIR)
    process_unmapped_repos()
