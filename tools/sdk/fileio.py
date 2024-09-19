import yaml 

def parse_spec(spec_file: str):
    """
    Parse the spec file and return the dictionary of specifications.
    Args:
        spec_file (str): Path to the spec file.
    Returns:
        dict: Dictionary of specifications.
    Raises:
        FileNotFoundError: If the spec file is not found.
    """

    # Load the spec file
    with open(spec_file, "r") as f:
        spec = yaml.safe_load(f)

    return spec