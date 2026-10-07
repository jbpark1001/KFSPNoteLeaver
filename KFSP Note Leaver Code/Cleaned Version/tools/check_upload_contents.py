"""Check that this snapshot contains only its approved code/template inventory."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parents[1]
manifest=json.loads((P/'UPLOAD_MANIFEST.json').read_text(encoding='utf-8'))
allowed=set(manifest)|{'UPLOAD_MANIFEST.json'}
actual={p.relative_to(P).as_posix() for p in P.rglob('*') if p.is_file()
        and '.git' not in p.relative_to(P).parts}
extra=sorted(actual-allowed)
missing=sorted(allowed-actual)
changed=[name for name,digest in manifest.items() if (P/name).is_file()
         and hashlib.sha256((P/name).read_bytes()).hexdigest()!=digest]
if extra or missing or changed:
    print(json.dumps({'unexpected_files':extra,'missing_files':missing,'changed_files':changed},indent=2))
    raise SystemExit('Snapshot differs from the reviewed upload inventory. Inspect before uploading.')
print(f'PASS: {len(actual)} approved files; no study artifacts or additional files present.')
