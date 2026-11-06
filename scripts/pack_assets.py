#!/usr/bin/env python3
"""Build, verify and import checksum-bound SWAAP asset bundles.

No Torch or simulator dependency. Builds accept an explicit source map, never
search a filesystem, and never modify source artifacts. Distribution profiles
may share content-addressed archive components without duplicating identical
bytes. Run this outside the source tree for large distribution artifacts.
"""
from __future__ import annotations
import argparse,hashlib,io,json,os,shutil,tarfile,tempfile
from pathlib import Path

SCHEMA=1

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p)
 with p.open('x')as f:json.dump(x,f,indent=2);f.write('\n')
def checked(p,a):
 p=Path(p)
 if not p.is_file()or p.stat().st_size!=a['bytes']or sha(p)!=a['sha256']:raise ValueError('Asset identity mismatch: '+a['id'])
 return p

def groups(manifest,profiles,max_bytes):
 entries={x['id']:x for x in manifest['assets']};bysha={}
 for name,p in profiles['profiles'].items():
  for ident in p['asset_ids']:
   if ident not in entries:raise ValueError('Unknown required asset: '+ident)
   a=entries[ident];o=bysha.setdefault(a['sha256'],dict(sha256=a['sha256'],bytes=a['bytes'],assets={},profiles=set()))
   if o['bytes']!=a['bytes']:raise ValueError('Conflicting size for digest')
   o['assets'][ident]=dict(id=ident,filename=a['filename'],sha256=a['sha256'],bytes=a['bytes']);o['profiles'].add(name)
 buckets={}
 for obj in bysha.values():buckets.setdefault(tuple(sorted(obj['profiles'])),[]).append(obj)
 result=[]
 for names,objects in sorted(buckets.items()):
  chunk=[];size=0;part=1
  for obj in sorted(objects,key=lambda x:x['sha256']):
   if chunk and size+obj['bytes']>max_bytes:result.append((names,part,chunk));chunk=[];size=0;part+=1
   chunk.append(obj);size+=obj['bytes']
  if chunk:result.append((names,part,chunk))
 return entries,result

class HashReader:
 def __init__(self,f):self.f=f;self.h=hashlib.sha256();self.n=0
 def read(self,n=-1):
  z=self.f.read(n);self.h.update(z);self.n+=len(z);return z

def info(name,size):
 t=tarfile.TarInfo(name);t.size=size;t.mtime=0;t.mode=0o444;t.uid=t.gid=0;t.uname=t.gname='';return t

def build(a):
 manifest,profiles,sources=read(a.manifest),read(a.profiles),read(a.sources)
 entries,components=groups(manifest,profiles,a.max_component_bytes);selected=set(a.only or profiles['profiles'])
 if selected-set(profiles['profiles']):raise ValueError('Unknown profile')
 sources=sources.get('sources',sources);bysha={}
 for key,val in sources.items():
  if isinstance(val,dict):val=val['path']
  bysha[entries[key]['sha256']if key in entries else key]=Path(val)
 chosen=[(n,i,o)for n,i,o in components if selected.intersection(n)]
 for _,_,objs in chosen:
  for o in objs:
   if o['sha256']not in bysha:raise ValueError('Missing explicit input: '+next(iter(o['assets'])))
   if not bysha[o['sha256']].is_file()or bysha[o['sha256']].stat().st_size!=o['bytes']:raise ValueError('Missing/wrong-size source: '+next(iter(o['assets'])))
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);results=[]
 for names,part,objs in chosen:
  name='__'.join(names)+f'.part{part:02d}.tar';tmp=out/(name+'.partial')
  aliases=[v for o in objs for v in o['assets'].values()]
  embedded=dict(schema_version=SCHEMA,assets=sorted(aliases,key=lambda x:x['id']))
  payload=(json.dumps(embedded,sort_keys=True,indent=2)+'\n').encode()
  with tarfile.open(tmp,'w',format=tarfile.PAX_FORMAT)as tar:
   tar.addfile(info('BUNDLE.json',len(payload)),io.BytesIO(payload))
   for o in objs:
    with bysha[o['sha256']].open('rb')as f:
     stream=HashReader(f);tar.addfile(info('objects/'+o['sha256'],o['bytes']),stream)
     if stream.n!=o['bytes']or stream.h.hexdigest()!=o['sha256']:raise ValueError('Source bytes changed or digest mismatched: '+next(iter(o['assets'])))
  target=out/name;tmp.rename(target)
  results.append(dict(file=name,sha256=sha(target),bytes=target.stat().st_size,profiles=list(names),asset_ids=sorted(x['id']for x in aliases),objects=[dict(sha256=o['sha256'],bytes=o['bytes'])for o in objs]))
  print(json.dumps(dict(completed_component=name,bytes=target.stat().st_size)),flush=True)
 result=dict(schema_version=SCHEMA,components=results,profiles={name:dict(description=profiles['profiles'][name]['description'],asset_ids=profiles['profiles'][name]['asset_ids'],archives=[r['file']for r in results if name in r['profiles']])for name in sorted(selected)},deduplication='Identical SHA-256 contents appear once across these archive components; a profile requires every archive in its list.',source_paths_included=False)
 if getattr(a,'notices',None):result['license_notices']=read(a.notices)
 save(out/'BUNDLE_MANIFEST.json',result)
 (out/'SHA256SUMS').write_text(''.join(r['sha256']+'  '+r['file']+'\n'for r in results))
 print(json.dumps(dict(status='BUILT',manifest=str(out/'BUNDLE_MANIFEST.json'),archive_bytes=sum(x['bytes']for x in results))))

def verify(a):
 m=read(a.bundle_manifest);root=Path(a.archive_root or Path(a.bundle_manifest).parent);selected=set(a.profile or m['profiles']);files=set(f for n in selected for f in m['profiles'][n]['archives']);objects=0
 for c in m['components']:
  if c['file']not in files:continue
  p=root/c['file']
  if p.stat().st_size!=c['bytes']or sha(p)!=c['sha256']:raise ValueError('Archive digest mismatch: '+c['file'])
  expected={x['sha256']:x['bytes']for x in c['objects']};seen=set()
  with tarfile.open(p,'r|')as t:
   for member in t:
    if member.name=='BUNDLE.json':continue
    digest=member.name.removeprefix('objects/')
    if not member.isfile()or member.name!='objects/'+digest or digest not in expected or member.size!=expected[digest]:raise ValueError('Unexpected archive member')
    stream=HashReader(t.extractfile(member))
    while stream.read(2**20):pass
    if stream.h.hexdigest()!=digest:raise ValueError('Object digest mismatch')
    seen.add(digest);objects+=1
  if seen!=set(expected):raise ValueError('Incomplete archive')
 print(json.dumps(dict(status='VERIFIED',archives=len(files),objects=objects,profiles=sorted(selected))))

def import_bundles(a):
 # Verify the complete requested profile before exposing any asset aliases.
 verify(a);m=read(a.bundle_manifest);archive_root=Path(a.archive_root or Path(a.bundle_manifest).parent);root=Path(a.asset_root);root.mkdir(parents=True,exist_ok=True)
 selected=set(a.profile or m['profiles']);ids=set(i for n in selected for i in m['profiles'][n]['asset_ids']);files=set(f for n in selected for f in m['profiles'][n]['archives']);aliases=[];cache=root/'.objects';cache.mkdir(exist_ok=True)
 for c in m['components']:
  if c['file']not in files:continue
  with tarfile.open(archive_root/c['file'],'r|')as t:
   for member in t:
    if member.name=='BUNDLE.json':aliases.extend(json.load(t.extractfile(member))['assets']);continue
    digest=member.name.removeprefix('objects/');dst=cache/digest
    if dst.exists():continue
    temp=cache/(digest+'.partial')
    with t.extractfile(member)as src,temp.open('xb')as out:shutil.copyfileobj(src,out,2**20)
    if sha(temp)!=digest:raise ValueError('Import digest mismatch')
    temp.rename(dst)
 count=0
 for alias in aliases:
  if alias['id']not in ids:continue
  rel=Path(alias['filename'])
  if rel.is_absolute()or'..'in rel.parts:raise ValueError('Unsafe asset filename')
  dst=root/rel;dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists():checked(dst,alias)
  else:
   try:os.link(cache/alias['sha256'],dst)
   except OSError:shutil.copyfile(cache/alias['sha256'],dst)
  count+=1
 print(json.dumps(dict(status='IMPORTED',assets=count,asset_root=str(root))))

def main():
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
 b=s.add_parser('build');b.add_argument('--manifest',required=True);b.add_argument('--profiles',required=True);b.add_argument('--sources',required=True);b.add_argument('--output',required=True);b.add_argument('--only',action='append');b.add_argument('--notices');b.add_argument('--max-component-bytes',type=int,default=1000000000)
 for name in ['verify','import']:
  q=s.add_parser(name);q.add_argument('--bundle-manifest',required=True);q.add_argument('--archive-root');q.add_argument('--profile',action='append')
  if name=='import':q.add_argument('--asset-root',required=True)
 a=p.parse_args();{'build':build,'verify':verify,'import':import_bundles}[a.command](a)
if __name__=='__main__':main()
