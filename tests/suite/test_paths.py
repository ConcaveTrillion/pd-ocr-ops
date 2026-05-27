from pdomain_ops.suite.paths import (
    installed_toml_path,
    jobs_db_path,
    suite_data_dir,
    ui_prefs_json_path,
)


def test_suite_data_dir_returns_path_and_creates(tmp_path, monkeypatch):
    monkeypatch.setenv("PD_SUITE_DATA_DIR", str(tmp_path))
    result = suite_data_dir()
    assert result == tmp_path
    assert result.exists()


def test_installed_toml_path_under_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PD_SUITE_DATA_DIR", str(tmp_path))
    result = installed_toml_path()
    assert result == tmp_path / "installed.toml"


def test_ui_prefs_json_path_under_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PD_SUITE_DATA_DIR", str(tmp_path))
    result = ui_prefs_json_path()
    assert result == tmp_path / "ui-prefs.json"


def test_jobs_db_path_under_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PD_SUITE_DATA_DIR", str(tmp_path))
    result = jobs_db_path()
    assert result == tmp_path / "jobs.db"
