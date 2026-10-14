import sys, json
sys.path.insert(0,".")
from v2 import lauf
part=int(sys.argv[1]); out=open(f"roh_{part}.jsonl","a")
aufg=[(ds,s) for s in range(920,940) for ds in ("mnist","fashion_mnist")]
for i,a in enumerate(aufg):
    if i%2==part:
        out.write(json.dumps(lauf(a))+"\n"); out.flush()
