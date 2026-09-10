import pytest

from backend import config
from backend import datasets as ds


@pytest.fixture
def dataset_root(tmp_path, monkeypatch):
    links = {}
    monkeypatch.setattr(ds, "datasets_root", lambda: tmp_path)
    monkeypatch.setattr(config, "datasets_root", lambda: tmp_path)
    monkeypatch.setattr(ds, "load_links", lambda: links)

    def save(new):
        links.clear()
        links.update(new)

    monkeypatch.setattr(ds, "save_links", save)
    return tmp_path
