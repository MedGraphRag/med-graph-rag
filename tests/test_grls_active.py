import asyncio
import hashlib
import json
from unittest.mock import patch

import httpx
import pytest

from src.nlp.grls_active import download_active_registry


def collect(tmp_path, *, missing_id=False, change_total=False, duplicate_id=False, unexpected_status=False):
    rows = [
        {'lpId':'a','commercialName':'Первый','mnnName':'Вещество','conditionId':'Д','atx':['C01']},
        {'lpId':'b','commercialName':'Второй','mnnName':'Вещество','conditionId':'Д','atx':[]},
        {'lpId':'c','commercialName':'Третий','mnnName':'Вещество','conditionId':'П','atx':['C02']},
    ]
    if unexpected_status:
        rows[1]['conditionId'] = 'ИЗМ'
    if missing_id:
        rows[1]['lpId'] = None
    if duplicate_id:
        rows[1]['lpId'] = 'a'
    calls = []
    controls = 0
    original_client = httpx.AsyncClient
    def handler(request):
        nonlocal controls
        calls.append(request)
        params = request.url.params
        assert params.get_list('conditionIds') == ['Д','П']
        assert params['page'] == '1'  # exact counts prevent out-of-range pagination
        if params['size'] == '1':
            controls += 1
            return httpx.Response(200, json=rows[:1], headers={'x-total-count':str(4 if controls==2 and change_total else 3)})
        if 'entryDateStart' not in params:
            result = rows[:1]
            assert params['entryDateEnd'].startswith('2024-01-01')
        elif 'entryDateEnd' in params:
            result = rows[1:2]
        else:
            result = rows[2:]
        return httpx.Response(200, json=result, headers={'x-total-count':'1'})
    def client(**options):
        return original_client(transport=httpx.MockTransport(handler), **options)
    with patch('httpx.AsyncClient', side_effect=client):
        snapshot = asyncio.run(download_active_registry(output_root=tmp_path, delay_seconds=0))
    return snapshot, calls


def test_collects_all_active_and_keeps_records_without_atx(tmp_path):
    snapshot, calls = collect(tmp_path)
    manifest = json.loads((snapshot/'manifest.json').read_text())
    assert len(calls) == 5
    assert manifest['complete'] and manifest['coverage_verified']
    assert not manifest['registry_complete']
    assert manifest['coverage_scope'] == 'conditionIds=Д,П'
    assert manifest['records_saved_unique'] == 3
    assert manifest['records_with_atx'] == 2
    assert manifest['condition_id_counts'] == {'Д':2, 'П':1}
    for origin in manifest['page_origins']:
        raw = (snapshot/'pages'/origin['page']).read_bytes()
        original = (tmp_path/origin['source_snapshot']/'pages'/origin['source_page']).read_bytes()
        assert raw == original
        assert hashlib.sha256(raw).hexdigest() == origin['sha256']


@pytest.mark.parametrize('options,message', [
    ({'missing_id':True}, 'Missing identity'),
    ({'unexpected_status':True}, 'Missing identity or unexpected status'),
    ({'change_total':True}, 'Coverage mismatch'),
    ({'duplicate_id':True}, 'Coverage mismatch'),
])
def test_does_not_publish_incomplete_collection(tmp_path, options, message):
    with pytest.raises(RuntimeError, match=message):
        collect(tmp_path, **options)
    assert not list(tmp_path.glob('snapshot_*'))
    report = json.loads(next(tmp_path.glob('collection_active_*/manifest.json')).read_text())
    assert not report['complete']
    assert report['termination_reason'] == 'error'
