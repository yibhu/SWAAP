import importlib.util
from pathlib import Path


def test_advertised_artifact_commands_and_configuration_inputs_resolve():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('check_recipes', root/'scripts/check_recipes.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.check(root)
    assert result['paper_rows'] == 35
    assert result['configurations'] >= 60
    assert result['parsed_artifact_commands'] >= 35
