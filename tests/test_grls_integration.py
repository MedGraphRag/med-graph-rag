import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import polars as pl
import pytest

from src.nlp.grls_scraper import download_registry
from src.nlp.grls_etl import normalize_snapshot, repair_mojibake


def download(tmp_path, payloads, **kwargs):
    calls = []
    def handler(request):
        calls.append(request)
        value = payloads[len(calls)-1]
        if isinstance(value, Exception):
            raise value
        if isinstance(value, httpx.Response):
            return value
        return httpx.Response(200, json=value)
    client_type = httpx.AsyncClient
    def client(**options):
        return client_type(transport=httpx.MockTransport(handler), **options)
    with patch('src.nlp.grls_scraper.httpx.AsyncClient', side_effect=client):
        result = asyncio.run(download_registry(output_root=tmp_path, delay_seconds=0, **kwargs))
    return result, calls


def test_pagination_and_filters(tmp_path):
    folder, calls = download(tmp_path, [[{'lpId': 'a'}], [{'lpId': 'b'}], []], condition_ids=['1', '2'])
    assert [r.url.params['page'] for r in calls] == ['1', '2', '3']
    assert calls[0].url.params.get_list('conditionIds') == ['1', '2']
    assert calls[0].url.params['size'] == '100'
    manifest = json.loads((folder/'manifest.json').read_text())
    assert manifest['pagination_complete'] and not manifest['complete']
    assert manifest['records_saved_unique'] == 2


@pytest.mark.parametrize('pages,reason,count', [
    ([[{'lpId': 'a'}, {'lpId': 'b'}], [{'lpId': 'b'}, {'lpId': 'a'}]], 'repeated_page', 2),
    ([[]], 'empty_page', 0),
    ([[{'lpId': 'a'}]], 'max_pages_reached', 1),
])
def test_termination(tmp_path, pages, reason, count):
    folder, calls = download(tmp_path, pages, max_pages=len(pages))
    manifest = json.loads((folder/'manifest.json').read_text())
    assert manifest['termination_reason'] == reason
    assert manifest['records_saved_unique'] == count
    assert 'conditionIds' not in calls[0].url.params


def test_invalid_json_preserved(tmp_path):
    raw = b'not json'
    with pytest.raises(RuntimeError, match='invalid JSON'):
        download(tmp_path, [httpx.Response(200, content=raw)])
    folder = next(tmp_path.glob('snapshot_*'))
    assert (folder/'pages/page_000001.json').read_bytes() == raw
    assert json.loads((folder/'manifest.json').read_text())['termination_reason'] == 'error'


@pytest.mark.parametrize('first', [httpx.Response(429, headers={'Retry-After':'0'}), httpx.Response(503, headers={'Retry-After':'0'}), httpx.ReadTimeout('timeout')])
def test_retries(tmp_path, first):
    folder, calls = download(tmp_path, [first, []])
    assert len(calls) == 2
    assert folder.exists()


def test_normalization_reproducible(tmp_path):
    snapshot = tmp_path/'snapshot'; snapshot.mkdir()
    pages = snapshot/'pages'; pages.mkdir()
    rows = []
    names = {'Эналаприл':'Эналаприл', 'Аторвастатин':'Аторвастатин', 'Бисопролол':'Бисопролол', 'Ренитек':'Эналаприл', 'Липримар':'Аторвастатин', 'Конкор':'Бисопролол'}
    for name, inn in names.items():
        rows.append({'lpId': name, 'commercialName': name+' таблетки (пленочные) 10 мг', 'mnnName': inn, 'mnnCode':['1','2'], 'atx':['C01','C02'], 'releaseForms':[{'name':'таблетки'}]})
    rows += [rows[0], {'lpId':'special','commercialName':'Тест капсулы 5 мг','mnnName':'~'}]
    raw = pages/'page_000001.json'; raw.write_text(json.dumps(rows, ensure_ascii=False))
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    output = tmp_path/'processed'/'result.parquet'
    with pytest.raises(RuntimeError, match='Missing manifest'):
        normalize_snapshot(snapshot, output)
    normalize_snapshot(snapshot, output, allow_incomplete=True)
    first = output.read_bytes()
    normalize_snapshot(snapshot, output, allow_incomplete=True)
    assert output.read_bytes() == first
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == digest
    df = pl.read_parquet(output)
    assert df.columns == ['trade_name_norm','inn_norm','atx_code']
    assert df.height == 7
    for name, inn in names.items():
        assert df.filter(pl.col('trade_name_norm') == name.lower())['inn_norm'][0] == inn.lower()
    assert df.filter(pl.col('trade_name_norm')=='тест')['inn_norm'][0] is None
    provenance = pl.read_parquet(tmp_path/'processed'/'result_records.parquet')
    assert provenance['mnn_codes'][0].to_list() == ['1','2']
    assert df.filter(pl.col('trade_name_norm')=='эналаприл')['atx_code'][0].to_list() == ['C01','C02']


@pytest.mark.parametrize('text', ['Цефуроксим','Зиннат®','Моксифлоксацин'])
def test_repair_idempotent(text):
    broken = text.encode('utf-8').decode('cp1251')
    assert repair_mojibake(broken) == text
    assert repair_mojibake(repair_mojibake(broken)) == text


def test_explicit_snapshot_and_bad_page(tmp_path):
    from src.nlp.grls_etl import find_latest_snapshot
    snapshot = tmp_path/'snapshot'; snapshot.mkdir()
    pages = snapshot/'pages'; pages.mkdir()
    (pages/'page_000001.json').write_text('broken')
    assert find_latest_snapshot(snapshot) == snapshot
    output = tmp_path/'processed'/'result.parquet'
    with pytest.raises(RuntimeError, match='invalid JSON'):
        normalize_snapshot(snapshot, output, allow_incomplete=True)
    assert not output.exists()


def test_output_cannot_modify_raw(tmp_path):
    with pytest.raises(ValueError, match='outside'):
        normalize_snapshot(tmp_path, tmp_path/'overwrite.parquet')


def test_unknown_schema(tmp_path):
    with pytest.raises(RuntimeError, match='Unrecognized response schema'):
        download(tmp_path, [{'unexpected': 'value'}])


def test_no_new_ids_and_duplicate_counts(tmp_path):
    folder, _ = download(tmp_path, [[{'lpId':'a'}, {'lpId':'b'}], [{'lpId':'a'}]])
    manifest = json.loads((folder/'manifest.json').read_text())
    assert manifest['termination_reason'] == 'no_new_records'
    assert manifest['records_received_including_duplicates'] == 3
    assert manifest['records_saved_unique'] == 2


def test_sparse_api_page_stops_before_more_requests(tmp_path):
    sparse = {'lpId': None, 'commercialName': None, 'mnnName': 'Вещество',
              'ruNumber': 'ЛП-123', 'regdata': {'regId': 'registration-only'}}
    folder, calls = download(tmp_path, [
        httpx.Response(200, json=[{'lpId':'a', 'commercialName':'Препарат'}],
                       headers={'x-total-count':'16585'}),
        httpx.Response(200, json=[sparse], headers={'x-total-count':'16585'}),
    ])
    manifest = json.loads((folder/'manifest.json').read_text())
    assert len(calls) == 2
    assert manifest['termination_reason'] == 'unusable_page'
    assert manifest['quality_issue']['page'] == 2
    assert not manifest['complete'] and not manifest['pagination_complete']
    assert manifest['api_total_count_initial'] == 16585
    assert manifest['records_received_including_duplicates'] == 2
    assert manifest['records_saved_unique'] == 1
    assert json.loads((folder/'pages/page_000002.json').read_text()) == [sparse]


def test_cancellation_records_interrupted_state(tmp_path):
    with patch('src.nlp.grls_scraper._get_with_retries', side_effect=asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(download_registry(output_root=tmp_path))
    manifest = json.loads(next(tmp_path.glob('snapshot_*/manifest.json')).read_text())
    assert manifest['termination_reason'] == 'interrupted'
    assert manifest['finished_at_utc']
    assert not manifest['complete']


def test_progress_manifest_is_written_during_download(tmp_path):
    client_type = httpx.AsyncClient
    calls = 0
    def handler(request):
        nonlocal calls
        calls += 1
        manifest = json.loads(next(tmp_path.glob('snapshot_*/manifest.json')).read_text())
        assert manifest['termination_reason'] == 'running'
        if calls == 1:
            return httpx.Response(200, json=[{'lpId':'a', 'commercialName':'Препарат'}])
        assert manifest['records_saved_unique'] == 1
        assert manifest['pages_fetched'] == 1
        return httpx.Response(200, json=[])
    def client(**options):
        return client_type(transport=httpx.MockTransport(handler), **options)
    with patch('src.nlp.grls_scraper.httpx.AsyncClient', side_effect=client):
        asyncio.run(download_registry(output_root=tmp_path, delay_seconds=0))
    assert calls == 2


def test_etl_rejects_sparse_pages_without_overwriting_output(tmp_path):
    snapshot = tmp_path/'snapshot'; pages = snapshot/'pages'; pages.mkdir(parents=True)
    page = pages/'page_000101.json'
    page.write_text(json.dumps([{'lpId':None, 'commercialName':None, 'ruNumber':'ЛП-123'}]))
    original = page.read_bytes()
    output = tmp_path/'result.parquet'; output.write_bytes(b'existing output')
    with pytest.raises(RuntimeError, match='all records lack lpId and commercialName'):
        normalize_snapshot(snapshot, output, allow_incomplete=True)
    assert output.read_bytes() == b'existing output'
    assert page.read_bytes() == original


def test_mapped_only_excludes_unusable_records_and_keeps_provenance(tmp_path):
    snapshot = tmp_path/'snapshot'; pages = snapshot/'pages'; pages.mkdir(parents=True)
    records = [
        {'lpId':'ok', 'commercialName':'Бренд таблетки 10 мг', 'mnnName':'Вещество', 'mnnCode':['1','2'], 'atx':['C01','C02']},
        {'lpId':'no-code', 'commercialName':'Другой', 'mnnName':'Вещество', 'atx':[]},
        {'lpId':'blank-code', 'commercialName':'Пустой', 'mnnName':'Вещество', 'atx':['  ','']},
        {'lpId':'no-inn', 'commercialName':'Тест', 'mnnName':'~', 'atx':['C01']},
        {'lpId':'dup', 'commercialName':'Бренд', 'mnnName':'Вещество', 'atx':['C01','C02']},
    ]
    (pages/'page_000001.json').write_text(json.dumps(records))
    sparse = pages/'page_000101.json'
    sparse.write_text(json.dumps([{'lpId':None, 'commercialName':None,'mnnName':'Вещество','ruNumber':'ЛП-123','atx':None}]))
    original = sparse.read_bytes()
    out = tmp_path/'result.parquet'
    # Partial source acceptance remains separately explicit.
    with pytest.raises(RuntimeError, match='Missing manifest'):
        normalize_snapshot(snapshot, out, mapped_only=True)
    count = normalize_snapshot(snapshot, out, mapped_only=True, allow_incomplete=True)
    assert count == 1
    df = pl.read_parquet(out)
    assert df.to_dicts() == [{'trade_name_norm':'бренд','inn_norm':'вещество','atx_code':['C01','C02']}]
    assert pl.read_parquet(tmp_path/'result_records.parquet').height == 5
    report = json.loads((tmp_path/'result_report.json').read_text())
    assert not report['source_complete']
    assert report['sparse_pages'] == 1
    assert report['excluded_records'] == 3
    assert report['total_excluded_records'] == 4
    assert report['skipped_pages'] == 1
    assert report['skipped_page_records'] == 1
    assert report['skipped_page_files'] == ['page_000101.json']
    assert report['missing_atx'] == 2
    assert report['eligible_records_before_dictionary_dedup'] == 2
    assert sparse.read_bytes() == original
    first = out.read_bytes()
    first_report = (tmp_path/'result_report.json').read_bytes()
    normalize_snapshot(snapshot, out, mapped_only=True, allow_incomplete=True)
    assert out.read_bytes() == first
    assert (tmp_path/'result_report.json').read_bytes() == first_report


def test_skip_pages_keeps_complete_mapping_on_mixed_page(tmp_path):
    snapshot = tmp_path/'snapshot'; pages = snapshot/'pages'; pages.mkdir(parents=True)
    (pages/'page_000001.json').write_text(json.dumps([
        {'lpId':'a','commercialName':'Полный','mnnName':'Алюминия фосфат','atx':['A02AB03']},
        {'lpId':'b','commercialName':'Без кода','mnnName':'Вещество','atx':None},
    ]))
    # A named page with no ATX is also skipped, not just the observed sparse shape.
    (pages/'page_000002.json').write_text(json.dumps([
        {'lpId':'c','commercialName':'Название','mnnName':'Вещество','atx':[]},
    ]))
    out = tmp_path/'result.parquet'
    assert normalize_snapshot(snapshot, out, mapped_only=True, allow_incomplete=True) == 1
    assert pl.read_parquet(out)['inn_norm'][0] == 'алюминия фосфат'
    report = json.loads((tmp_path/'result_report.json').read_text())
    assert report['skipped_page_files'] == ['page_000002.json']
    assert report['total_excluded_records'] == 2


def test_all_pages_skipped_does_not_overwrite_existing_output(tmp_path):
    snapshot = tmp_path/'snapshot'; pages = snapshot/'pages'; pages.mkdir(parents=True)
    (pages/'page_000001.json').write_text(json.dumps([{'lpId':None,'commercialName':None,'ruNumber':'123'}]))
    out = tmp_path/'result.parquet'; out.write_bytes(b'previous')
    with pytest.raises(RuntimeError, match='No usable records remain'):
        normalize_snapshot(snapshot, out, mapped_only=True, allow_incomplete=True)
    assert out.read_bytes() == b'previous'


def test_merge_snapshots_preserves_links_and_source_evidence(tmp_path):
    from src.nlp.grls_etl import merge_snapshots
    old = tmp_path/'old'; new = tmp_path/'new'
    for directory, records in [
        (old, [{'lpId':'same', 'commercialName':'Бренд', 'mnnName':'Вещество', 'atx':['C01'], 'conditionId':'ИЗМ'}]),
        (new, [
            {'lpId':'same', 'commercialName':'Бренд', 'mnnName':'Вещество', 'atx':['C02'], 'conditionId':'Д'},
            {'lpId':'new', 'commercialName':'Новый', 'mnnName':'Вещество', 'atx':['C03'], 'conditionId':'П'},
        ]),
    ]:
        pages = directory/'pages'; pages.mkdir(parents=True)
        (pages/'page_000001.json').write_text(json.dumps(records))
    out = tmp_path/'result.parquet'
    assert merge_snapshots([old,new,old], out, mapped_only=True, allow_incomplete=True) == 3
    df = pl.read_parquet(out)
    assert df['trade_name_norm'].to_list() == ['бренд','бренд','новый']
    records = pl.read_parquet(tmp_path/'result_records.parquet')
    assert records.height == 3
    assert set(records['source_snapshot'].to_list()) == {str(old.resolve()),str(new.resolve())}
    assert records.filter(pl.col('lp_id')=='same')['condition_id'].to_list() == ['ИЗМ','Д']
    assert set(records['source_page'].to_list()) == {'page_000001.json'}
    report = json.loads((tmp_path/'result_report.json').read_text())
    assert report['new_links_added'] == 2 and not report['source_complete']
    first = out.read_bytes(); first_report = (tmp_path/'result_report.json').read_bytes()
    merge_snapshots([old,new], out, mapped_only=True, allow_incomplete=True)
    assert out.read_bytes() == first
    assert (tmp_path/'result_report.json').read_bytes() == first_report


def test_merge_invalid_source_does_not_overwrite_output(tmp_path):
    from src.nlp.grls_etl import merge_snapshots
    for name, content in [('old',json.dumps([{'lpId':'a','commercialName':'Бренд','mnnName':'МНН','atx':['C01']}])),('broken','invalid')]:
        pages = tmp_path/name/'pages'; pages.mkdir(parents=True)
        (pages/'page_000001.json').write_text(content)
    output = tmp_path/'result.parquet'; output.write_bytes(b'previous')
    with pytest.raises(RuntimeError, match='invalid JSON'):
        merge_snapshots([tmp_path/'old',tmp_path/'broken'],output,mapped_only=True,allow_incomplete=True)
    assert output.read_bytes() == b'previous'


def test_stops_at_stable_reported_total_and_passes_date_filter(tmp_path):
    folder,calls = download(tmp_path,[httpx.Response(200,json=[{'lpId':'a','commercialName':'Бренд'}],headers={'x-total-count':'1'})],entry_date_start='2024-01-01',entry_date_end='2025-01-01')
    assert len(calls)==1
    assert calls[0].url.params['entryDateStart']=='2024-01-01T00:00:00.000Z'
    assert calls[0].url.params['entryDateEnd']=='2025-01-01T00:00:00.000Z'
    manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['query_complete']
    assert manifest['termination_reason']=='reported_total_reached'
    assert not manifest['complete']


def test_detects_total_count_drift(tmp_path):
    folder,_ = download(tmp_path,[httpx.Response(200,json=[{'lpId':'a'}],headers={'x-total-count':'3'}),httpx.Response(200,json=[{'lpId':'b'}],headers={'x-total-count':'4'})])
    manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['termination_reason']=='total_count_changed'
    assert not manifest['query_complete']
