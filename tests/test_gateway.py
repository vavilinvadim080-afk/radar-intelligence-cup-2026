import base64
import json
import io
import zipfile
import pytest
from radar_app.gateway import InternalGateway

@pytest.mark.parametrize("url", ["https://api.openai.com/v1", "https://8.8.8.8/v1", "http://10.0.0.3/v1", "http://169.254.169.254/v1", "file:///tmp/model", "http://user:pass@127.0.0.1/v1", "http://127.0.0.1/v1?url=https://evil.org"])
def test_external_metadata_credentials_and_cleartext_nonlocal_denied(url):
    with pytest.raises(ValueError):
        InternalGateway(url, allowed_hosts=["127.0.0.1","10.0.0.3","8.8.8.8","169.254.169.254","api.openai.com"])

def test_internal_ip_requires_explicit_allowlist():
    with pytest.raises(ValueError):
        InternalGateway("https://10.2.3.4/v1",allowed_hosts=["10.2.3.5"])

def test_structured_model_contract_preserves_trust_boundary():
    calls=[]
    def transport(path,body,headers):
        calls.append((path,json.loads(body)))
        a=dict(topic_key="export-orders",label="Экспорт",kind="wish",quote="Хочу Excel",confidence=.9)
        return json.dumps({"choices":[{"message":{"content":json.dumps(a)}}]}).encode()
    g=InternalGateway("http://127.0.0.1:11434/v1",allowed_hosts=["127.0.0.1"],transport=transport)
    a=g.analyze("Хочу Excel",[])
    assert a["kind"]=="wish"
    assert calls[0][0]=="/chat/completions"
    body=calls[0][1]
    assert body["response_format"]["type"]=="json_schema"
    assert body["messages"][0]["role"]=="system"
    assert "Хочу Excel" not in body["messages"][0]["content"]
    assert json.loads(body["messages"][1]["content"])["ticket"]=="Хочу Excel"

def test_model_cannot_fabricate_quote_even_with_valid_json():
    def transport(*args):
        a=dict(topic_key="export",label="Экспорт",kind="wish",quote="НЕ БЫЛО",confidence=.9)
        return json.dumps({"choices":[{"message":{"content":json.dumps(a)}}]}).encode()
    g=InternalGateway("http://127.0.0.1/v1",allowed_hosts=["127.0.0.1"],transport=transport)
    with pytest.raises(ValueError):
        g.analyze("Хочу Excel",[])

def test_audio_uploaded_as_bytes_not_remote_url_and_cached():
    calls=[]
    def transport(path,body,headers):
        calls.append((path,body,headers));return b'{"text":"Please export orders"}'
    g=InternalGateway("http://127.0.0.1/v1",allowed_hosts=["127.0.0.1"],transport=transport)
    t={"media_b64":base64.b64encode(b'RIFFxxxxWAVEfmt ').decode(),"mime":"audio/wav"}
    assert g.extract("audio",t)=="Please export orders"
    assert g.extract("audio",t)=="Please export orders"
    assert len(calls)==1
    assert calls[0][0]=="/audio/transcriptions"
    assert b"RIFFxxxxWAVE" in calls[0][1]
    assert "multipart/form-data" in calls[0][2]["Content-Type"]

def test_image_uses_inline_data_and_vl_model():
    calls=[]
    def transport(path,body,headers):
        calls.append(json.loads(body));return b'{"choices":[{"message":{"content":"visible text"}}]}'
    g=InternalGateway("http://127.0.0.1/v1",allowed_hosts=["127.0.0.1"],transport=transport)
    t={"media_b64":base64.b64encode(b'\x89PNG\r\n\x1a\nfixture').decode(),"mime":"image/png"}
    assert g.extract("image",t)=="visible text"
    assert calls[0]["model"]=="Qwen-VL"
    assert calls[0]["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")

def test_rejects_remote_media_and_wrong_signature():
    g=InternalGateway("http://127.0.0.1/v1",allowed_hosts=["127.0.0.1"],transport=lambda *a:pytest.fail("Must not send"))
    with pytest.raises(ValueError):
        g.extract("image",{"url":"https://evil.org/img","mime":"image/png"})
    with pytest.raises(ValueError):
        g.extract("image",{"media_b64":base64.b64encode(b'<script>').decode(),"mime":"image/png"})

def test_model_budget_stops_excess_calls():
    a=dict(topic_key="x",label="X",kind="unknown",quote="x",confidence=0)
    response=json.dumps({"choices":[{"message":{"content":json.dumps(a)}}]}).encode()
    g=InternalGateway("http://127.0.0.1/v1",allowed_hosts=["127.0.0.1"],transport=lambda *args:response,max_calls=2)
    g.analyze("x",[]);g.analyze("x",[])
    with pytest.raises(RuntimeError):
        g.analyze("x",[])

def test_document_text_and_docx_are_read_without_model_call():
    g=InternalGateway('http://127.0.0.1/v1',allowed_hosts=['127.0.0.1'],transport=lambda *a:pytest.fail('No model needed'))
    assert g.extract('document',{'mime':'text/plain','media_b64':base64.b64encode('Нужен экспорт заказов'.encode()).decode()})=='Нужен экспорт заказов'
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:
        z.writestr('word/document.xml','<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Нужен Excel</w:t></w:r></w:p></w:body></w:document>')
    assert 'Нужен Excel' in g.extract('document',{'mime':'application/vnd.openxmlformats-officedocument.wordprocessingml.document','media_b64':base64.b64encode(b.getvalue()).decode()})

def test_document_entity_and_macro_content_rejected():
    g=InternalGateway('http://127.0.0.1/v1',allowed_hosts=['127.0.0.1'])
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:
        z.writestr('word/document.xml','<!DOCTYPE x [<!ENTITY x "unsafe">]><x>&x;</x>')
    with pytest.raises(ValueError):g.extract('document',{'mime':'application/vnd.openxmlformats-officedocument.wordprocessingml.document','media_b64':base64.b64encode(b.getvalue()).decode()})
