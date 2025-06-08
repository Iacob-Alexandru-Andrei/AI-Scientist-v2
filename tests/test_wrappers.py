import types
from ai_scientist.paper_improver import llm_review, vlm_review


class DummyClient:
    pass


def test_llm_review_model(monkeypatch):
    called = {}

    def fake_create(model):
        called['model'] = model
        return DummyClient(), model

    def fake_perform(path, m, client, **k):
        called['perform'] = (m, path)
        return {'ok': True}

    monkeypatch.setattr(llm_review, 'create_client', fake_create)
    monkeypatch.setattr(llm_review, 'perform_review', fake_perform)

    res = llm_review.llm_review('file.pdf', model='some-model')
    assert res == {'ok': True}
    assert called['model'] == 'some-model'
    assert called['perform'] == ('some-model', 'file.pdf')


def test_vlm_review_model(monkeypatch):
    called = {}

    def fake_create(model):
        called['model'] = model
        return DummyClient(), model

    def fake_perform(client, model, path):
        called['perform'] = (model, path)
        return {'done': True}

    monkeypatch.setattr(vlm_review, 'create_vlm_client', fake_create)
    monkeypatch.setattr(vlm_review, 'perform_imgs_cap_ref_review', fake_perform)

    res = vlm_review.vlm_review('doc.pdf', model='vlm-model')
    assert res == {'done': True}
    assert called['model'] == 'vlm-model'
    assert called['perform'] == ('vlm-model', 'doc.pdf')
