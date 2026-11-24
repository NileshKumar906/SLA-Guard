from pathlib import Path
import zipfile, requests, pandas as pd
from .config import RAW
UCI_PAGE="https://archive.ics.uci.edu/dataset/498/incident%2Bmanagement%2Bprocess%2Benriched%2Bevent%2Blog"
def download():
 RAW.mkdir(parents=True,exist_ok=True); url="https://archive.ics.uci.edu/static/public/498/incident+management+process+enriched+event+log.zip"; zpath=RAW/"uci_dataset.zip"
 try:
  r=requests.get(url,timeout=60,headers={"User-Agent":"SLA-Guard"}); r.raise_for_status()
  if not r.content.startswith(b"PK"): raise ValueError("Expected ZIP archive")
  zpath.write_bytes(r.content); zipfile.ZipFile(zpath).extractall(RAW); return {"status":"downloaded","path":str(zpath)}
 except Exception as e: return {"status":"failed","error":str(e),"manual_source":UCI_PAGE}
def locate():
 files=[p for p in RAW.rglob("*") if p.is_file() and p.suffix.lower() in (".csv",".xlsx",".xls")]
 if not files: raise FileNotFoundError("Place the downloaded CSV/XLSX in data/raw")
 files.sort(key=lambda p:(p.suffix.lower()!=".csv",len(str(p)))); return files[0]
def load(path=None):
 p=Path(path) if path else locate()
 df=pd.read_csv(p,low_memory=False) if p.suffix.lower()==".csv" else pd.read_excel(p)
 df.columns=[str(c).strip() for c in df.columns]; return df,p
def audit(df):
 return {"rows":len(df),"columns":len(df.columns),"column_names":list(df.columns),"missing_percent":(df.isna().mean()*100).round(2).to_dict(),"duplicate_rows":int(df.duplicated().sum()),"target_candidates":[c for c in df if c.lower() in ["made_sla","sla","sla_met","breached_sla"]],"timestamp_candidates":[c for c in df if any(x in c.lower() for x in ["opened","created","resolved","closed","time"])],"preview":df.head(5).astype(str).to_dict("records")}
