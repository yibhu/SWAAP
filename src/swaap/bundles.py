"""Verified local/public archive acquisition for the named reproduction bundles.

The manifest describes content-addressed, deduplicated archive components. No
models are executed and no source data are changed. A local bundle directory is
usable before an author selects a public hosting endpoint.
"""
from __future__ import annotations
import hashlib,json,os,shutil,tarfile,urllib.parse,urllib.request
from pathlib import Path
from .artifacts import AssetStore,AssetError

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for data in iter(lambda:f.read(2**20),b''):h.update(data)
 return h.hexdigest()

def safe_relative(value):
 p=Path(value)
 if p.is_absolute()or'..'in p.parts or '\\'in value or not p.parts:raise AssetError('Unsafe bundle filename: '+value)
 return p

class BundleStore:
 def __init__(self,root='artifacts',bundle_dir='bundles',manifest=None,asset_manifest=None):
  self.root=Path(root).expanduser().resolve();self.bundle_dir=Path(bundle_dir).expanduser().resolve()
  default=Path(__file__).with_name('resources')/'bundle_manifest.json'
  checkout=Path(__file__).resolve().parents[2]/'assets/bundle_manifest.json'
  self.manifest_path=Path(manifest)if manifest else(checkout if checkout.exists()else default)
  if not self.manifest_path.is_file():raise AssetError('Bundle manifest is missing; specify --manifest PATH.')
  self.manifest=json.loads(self.manifest_path.read_text());self.components={c['file']:c for c in self.manifest['components']};self.profiles=self.manifest['profiles'];self.assets=AssetStore(self.root,asset_manifest)
  for c in self.components.values():safe_relative(c['file'])
 def _required(self,profile):
  if profile not in self.profiles:raise AssetError('Unknown bundle profile: '+str(profile))
  p=self.profiles[profile];components=[]
  for name in p['archives']:
   if name not in self.components:raise AssetError('Unknown archive dependency: '+name)
   components.append(self.components[name])
  return p,components
 def list_profiles(self):
  return[dict(profile=name,description=p['description'],assets=len(p['asset_ids']),archives=len(p['archives']),archive_bytes=sum(self.components[f]['bytes']for f in p['archives']),local_archives_present=sum((self.bundle_dir/f).is_file()for f in p['archives']))for name,p in sorted(self.profiles.items())]
 def verify(self,profile):
  p,components=self._required(profile);available=set();objects=set()
  for c in components:
   file=self.bundle_dir/c['file']
   if not file.is_file():raise AssetError('Missing bundle archive '+c['file']+'; place the prepared archive in --bundle-dir.')
   if file.stat().st_size!=c['bytes']or digest(file)!=c['sha256']:raise AssetError('Bundle archive identity mismatch: '+c['file'])
   expected={x['sha256']:x['bytes']for x in c['objects']};seen=set();alias_ids=[];bundle_seen=False
   with tarfile.open(file,'r|')as tar:
    for member in tar:
     if not member.isfile():raise AssetError('Only regular files are permitted in bundle archives.')
     if member.name=='BUNDLE.json':
      if bundle_seen:raise AssetError('Duplicate BUNDLE.json member.')
      bundle_seen=True
      embedded=json.load(tar.extractfile(member));alias_ids=[a['id']for a in embedded['assets']]
      if len(alias_ids)!=len(set(alias_ids)):raise AssetError('Duplicate bundle alias IDs.')
      for a in embedded['assets']:
       safe_relative(a['filename']);canon=self.assets.entries.get(a['id'])
       if a['sha256']not in expected or expected[a['sha256']]!=a['bytes']:raise AssetError('Alias has no matching content object: '+a['id'])
       if canon is None or any(canon[k]!=a[k]for k in ['filename','sha256','bytes']):raise AssetError('Bundle alias differs from canonical asset: '+a['id'])
      available.update(alias_ids);continue
     sha=member.name.removeprefix('objects/')
     if member.name!='objects/'+sha or sha not in expected or member.size!=expected[sha]or sha in seen:raise AssetError('Unexpected bundle member: '+member.name)
     h=hashlib.sha256()
     with tar.extractfile(member)as f:
      for data in iter(lambda:f.read(2**20),b''):h.update(data)
     if h.hexdigest()!=sha:raise AssetError('Bundle object digest mismatch: '+sha)
     seen.add(sha);objects.add(sha)
   if not bundle_seen or seen!=set(expected)or set(alias_ids)!=set(c['asset_ids']):raise AssetError('Incomplete archive manifest: '+c['file'])
  if not set(p['asset_ids'])<=available:raise AssetError('Bundle profile has unresolved required assets.')
  return dict(status='VERIFIED',profile=profile,archives=len(components),unique_objects=len(objects),assets=len(p['asset_ids']))
 def install(self,profile):
  result=self.verify(profile);p,components=self._required(profile);wanted=set(p['asset_ids']);self.root.mkdir(parents=True,exist_ok=True);cache=self.root/'.objects';cache.mkdir(exist_ok=True);aliases={}
  for c in components:
   with tarfile.open(self.bundle_dir/c['file'],'r|')as tar:
    for member in tar:
     if member.name=='BUNDLE.json':
      aliases.update({a['id']:a for a in json.load(tar.extractfile(member))['assets']if a['id']in wanted});continue
     sha=member.name.removeprefix('objects/');target=cache/sha
     if target.exists():
      if target.stat().st_size!=member.size or digest(target)!=sha:raise AssetError('Existing cached object differs: '+sha)
      continue
     temp=cache/(sha+'.partial')
     if temp.exists():raise AssetError('A previous partial import is preserved: '+str(temp))
     with tar.extractfile(member)as source,temp.open('xb')as out:shutil.copyfileobj(source,out,2**20)
     if digest(temp)!=sha:raise AssetError('Imported object differs: '+sha)
     temp.rename(target)
  if self.manifest.get('license_notices'):
   notice=(json.dumps(self.manifest['license_notices'],indent=2,sort_keys=True)+'\n').encode()
   directory=self.root/'licenses';directory.mkdir(exist_ok=True)
   path=directory/('notices-'+hashlib.sha256(notice).hexdigest()[:16]+'.json')
   if not path.exists():
    with path.open('xb')as f:f.write(notice)
  for ident,a in aliases.items():
   dst=self.assets.destination(ident);dst.parent.mkdir(parents=True,exist_ok=True)
   if dst.exists():self.assets.verify(ident)
   else:
    try:os.link(cache/a['sha256'],dst)
    except OSError:shutil.copyfile(cache/a['sha256'],dst)
   self.assets.verify(ident)
  return dict(result,status='INSTALLED',asset_root=str(self.root))
 def download(self,profile):
  _,components=self._required(profile);self.bundle_dir.mkdir(parents=True,exist_ok=True)
  for c in components:
   dst=self.bundle_dir/c['file']
   if dst.is_file()and dst.stat().st_size==c['bytes']and digest(dst)==c['sha256']:continue
   urls=c.get('urls',[])
   if not urls:raise AssetError('No public hosting URL is assigned for '+c['file']+'. Local archive verification/install works with --bundle-dir; do not substitute an unverified download.')
   url=urls[0]
   if urllib.parse.urlparse(url).scheme!='https':raise AssetError('Bundle downloads require HTTPS.')
   temp=dst.with_suffix(dst.suffix+'.partial')
   if temp.exists():raise AssetError('A previous partial download is preserved: '+str(temp))
   count=0
   with urllib.request.urlopen(url,timeout=60)as source,temp.open('xb')as out:
    if urllib.parse.urlparse(source.geturl()).scheme!='https':raise AssetError('Bundle redirect must retain HTTPS.')
    while True:
     data=source.read(min(2**20,c['bytes']-count+1))
     if not data:break
     count+=len(data)
     if count>c['bytes']:raise AssetError('Bundle download exceeded its declared byte limit; partial file preserved.')
     out.write(data)
   if temp.stat().st_size!=c['bytes']or digest(temp)!=c['sha256']:raise AssetError('Downloaded bundle identity mismatch: '+c['file'])
   temp.rename(dst)
  return self.verify(profile)
