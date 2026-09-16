"""One-time local spatial-only export. Never imported by request handling.

python -m dashboard.build_runtime_registry SOURCE.jsonl OUTPUT.json
No downloads, model calls, questions, answers or task labels are exported.
"""
import hashlib
import json
import sys
from pathlib import Path
from spatial.context_parser import parse_context
from spatial.schema import CATEGORIES


def build(source, output):
    blocks={}
    with Path(source).open() as handle:
        for line in handle:
            context=parse_context(json.loads(line).get('context',''))
            rows=[(p.name,p.category,p.latitude,p.longitude) for p in context.candidates]
            if context.errors or not rows:continue
            if any(c not in CATEGORIES or not n.strip() or len(n)>160 or any(ord(x)<32 for x in n) for n,c,_,_ in rows):continue
            key=hashlib.sha256(json.dumps(rows,ensure_ascii=False).encode()).hexdigest()
            blocks[key]=rows
    entities=[];seen=set();selected=0
    # Choose whole non-overlapping registry snapshots. Never merge same-name places
    # or collapse distinct ordinals inside a snapshot. IDs refer to source records,
    # not asserted OSM IDs (the local export does not retain those).
    for key,rows in sorted(blocks.items()):
        if any(row in seen for row in rows):continue
        for index,(name,category,lat,lon) in enumerate(rows):
            entities.append(dict(id=f'local-osm:{key[:20]}:{index}',name_ar=name,category=category,
                                 latitude=lat,longitude=lon,source='OpenStreetMap via local spatial export'))
        seen.update(rows);selected+=1
        if len(entities)>=500:break
    if not entities:raise ValueError('No eligible spatial records')
    Path(output).write_text(json.dumps(dict(version=1,description='Local OSM-derived spatial export; static, not live or completeness-verified',
        identity_policy='Source spatial-snapshot digest plus ordinal; no name-based deduplication',
        snapshot_count=selected,entities=entities),ensure_ascii=False,indent=2)+'\n')
    return len(entities)

if __name__=='__main__':print('Exported entities:',build(*sys.argv[1:]))
