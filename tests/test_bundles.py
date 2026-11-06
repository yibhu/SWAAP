import hashlib,importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from swaap.bundles import BundleStore
from swaap.artifacts import AssetError

ROOT=Path(__file__).resolve().parents[1]
def setup_bundle(tmp_path):
 spec=importlib.util.spec_from_file_location('swaap_public_packer',ROOT/'scripts/pack_assets.py');packer=importlib.util.module_from_spec(spec);spec.loader.exec_module(packer)
 one=tmp_path/'one';one.write_bytes(b'exact immutable model fixture\x00\x01');two=tmp_path/'two';two.write_bytes(b'separate model fixture')
 def entry(id,file):return dict(id=id,filename=id+'/artifact.pt',sha256=hashlib.sha256(file.read_bytes()).hexdigest(),bytes=file.stat().st_size,urls=[])
 manifest={'assets':[entry('first',one),entry('alias',one),entry('second',two)]};mp=tmp_path/'assets.json';mp.write_text(json.dumps(manifest));profiles=tmp_path/'profiles.json';profiles.write_text(json.dumps({'profiles':{'minimal':{'description':'one real fixture','asset_ids':['first']},'main':{'description':'two content objects','asset_ids':['alias','second']}}}));sources=tmp_path/'sources.json';sources.write_text(json.dumps({'sources':{'first':str(one),'alias':str(one),'second':str(two)}}));out=tmp_path/'bundles'
 notices=tmp_path/'notices.json';notices.write_text(json.dumps({'notices':[{'scope':'test fixture','text':'Retain this fixture notice.'}]}))
 packer.build(SimpleNamespace(manifest=str(mp),profiles=str(profiles),sources=str(sources),output=str(out),max_component_bytes=1000000,only=None,notices=str(notices)))
 return BundleStore(tmp_path/'installed',out,out/'BUNDLE_MANIFEST.json',mp),out,manifest

def test_bundle_dependency_dedup_and_minimal_install(tmp_path):
 store,out,assets=setup_bundle(tmp_path);m=json.loads((out/'BUNDLE_MANIFEST.json').read_text());objects=[x['sha256']for c in m['components']for x in c['objects']];assert len(objects)==len(set(objects))==2
 assert store.verify('minimal')['unique_objects']==1
 assert store.install('minimal')['assets']==1
 assert store.assets.destination('first').is_file()
 notice_files=list((store.root/'licenses').glob('notices-*.json'));assert len(notice_files)==1
 assert 'Retain this fixture notice.'in notice_files[0].read_text()
 assert not store.assets.destination('second').exists()
 assert store.install('main')['assets']==2
 assert store.assets.destination('alias').read_bytes()==store.assets.destination('first').read_bytes()
 assert store.verify('main')['unique_objects']==2

def test_bundle_tampering_and_unknown_profile_rejected(tmp_path):
 store,out,_=setup_bundle(tmp_path)
 with pytest.raises(AssetError,match='Unknown bundle profile'):store.verify('missing')
 component=json.loads((out/'BUNDLE_MANIFEST.json').read_text())['components'][0]['file'];p=out/component
 with p.open('r+b')as f:f.write(b'tampered')
 profile=next(x['profile']for x in store.list_profiles()if component in store.profiles[x['profile']]['archives'])
 with pytest.raises(AssetError,match='identity mismatch'):store.install(profile)
 assert not store.root.exists()

def test_bundle_download_requires_real_endpoint(tmp_path):
 store,out,_=setup_bundle(tmp_path)
 c=json.loads((out/'BUNDLE_MANIFEST.json').read_text())['components'][0]
 profile=c['profiles'][0];(out/c['file']).unlink()
 with pytest.raises(AssetError,match='No public hosting URL'):store.download(profile)

def rewrite_component(store,out,change):
 import io,tarfile
 c=store.components[next(iter(store.components))];p=out/c['file'];items=[]
 with tarfile.open(p)as t:
  for member in t:items.append((member.name,t.extractfile(member).read()))
 items=change(items)
 with tarfile.open(p,'w')as t:
  for name,data in items:
   member=tarfile.TarInfo(name);member.size=len(data);t.addfile(member,io.BytesIO(data))
 c['bytes']=p.stat().st_size;c['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 return c['profiles'][0]

def test_duplicate_bundle_metadata_and_traversal_are_rejected(tmp_path):
 store,out,_=setup_bundle(tmp_path)
 profile=rewrite_component(store,out,lambda items:items+[next(x for x in items if x[0]=='BUNDLE.json')])
 with pytest.raises(AssetError,match='Duplicate BUNDLE'):store.verify(profile)

 def traversal(items):
  result=[]
  for name,data in items:
   if name=='BUNDLE.json':
    b=json.loads(data);b['assets'][0]['filename']='../escaped.pt';data=json.dumps(b).encode()
   result.append((name,data))
  return result
 # Reuse another finite fixture directory without touching the rejected archive.
 fresh=tmp_path/'fresh';fresh.mkdir();store,out,_=setup_bundle(fresh)
 profile=rewrite_component(store,out,traversal)
 with pytest.raises(AssetError,match='Unsafe bundle filename'):store.verify(profile)

 def duplicate_alias(items):
  result=[]
  for name,data in items:
   if name=='BUNDLE.json':
    b=json.loads(data);b['assets'].append(dict(b['assets'][0]));data=json.dumps(b).encode()
   result.append((name,data))
  return result
 fresh=tmp_path/'duplicate-alias';fresh.mkdir();store,out,_=setup_bundle(fresh)
 profile=rewrite_component(store,out,duplicate_alias)
 with pytest.raises(AssetError,match='Duplicate bundle alias IDs'):store.verify(profile)

def test_missing_dependency_and_preserved_partial_import(tmp_path):
 store,out,_=setup_bundle(tmp_path)
 store.profiles['minimal']['archives'].append('missing-dependency.tar')
 with pytest.raises(AssetError,match='Unknown archive dependency'):store.verify('minimal')
 store.profiles['minimal']['archives'].pop()
 component=next(c for c in store.components.values()if'minimal'in c['profiles']);sha=component['objects'][0]['sha256'];cache=store.root/'.objects';cache.mkdir(parents=True);partial=cache/(sha+'.partial');partial.write_bytes(b'preserve failed partial')
 with pytest.raises(AssetError,match='partial import is preserved'):store.install('minimal')
 assert partial.read_bytes()==b'preserve failed partial'
 assert not store.assets.destination('first').exists()

def test_bundle_download_enforces_byte_limit(tmp_path,monkeypatch):
 import io
 store,out,_=setup_bundle(tmp_path);c=next(c for c in store.components.values()if'minimal'in c['profiles']);(out/c['file']).unlink();c['urls']=['https://example.invalid/published-archive']
 class Response(io.BytesIO):
  def geturl(self):return 'https://example.invalid/published-archive'
 monkeypatch.setattr('urllib.request.urlopen',lambda *args,**kwargs:Response(b'x'*(c['bytes']+1)))
 with pytest.raises(AssetError,match='declared byte limit'):store.download('minimal')
 partial=(out/c['file']).with_suffix('.tar.partial');assert partial.exists();before=partial.read_bytes()
 with pytest.raises(AssetError,match='partial download is preserved'):store.download('minimal')
 assert partial.read_bytes()==before
