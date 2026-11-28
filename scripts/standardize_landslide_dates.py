"""Conservative temporal sidecar for regional inventories; no label approval.
Run from project scripts directory. Inputs read-only; no geometry changes.
"""
import argparse
import calendar
import csv
import json
import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
MISSING = {'', 'na', 'n/a', 'nan', 'none', 'null', 'nat', 'unknown', 'nil', '-', '--', '<na>', '.'}
MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if i}
MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if i})
IST = timezone(timedelta(hours=5, minutes=30))
FIELDS = ['source','layer','qa_record_id','date_role','date_status','date_precision',
          'start_date','end_date','interval_days','raw_start','raw_end','raw_time',
          'encoded_timestamp_utc','alternative_ist_day','sat_before_raw','sat_after_raw',
          'sat_before','sat_after','sat_window_status','survey_timestamp_raw','survey_date_raw',
          'survey_date','survey_status','context_start','context_end','representation_status']

def require(ok, msg):
    if not ok: raise ValueError(msg)

def missing(x): return str(x).strip().casefold() in MISSING

def iso(x): return x.isoformat() if x else ''

def dmy(y,m,d): return date(int(y),int(m),int(d))

def result(status, precision='UNKNOWN', a=None,b=None):
    return dict(date_status=status,date_precision=precision,start_date=iso(a),end_date=iso(b),interval_days=(b-a).days+1 if a and b else '')

def window(a,b,precision,as_of,status=None):
    if a>b: return result('REVERSED_INTERVAL')
    if b>as_of: return result('FUTURE_DATE_REVIEW')
    return result(status or ('DAY_CANDIDATE_UNVERIFIED' if a==b else 'INTERVAL_CANDIDATE_UNVERIFIED'),precision,a,b)

def strict_iso(s):
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',s): raise ValueError('not ISO date')
    return date.fromisoformat(s)

def uglc(a,b,as_of):
    a,b=a.strip(),b.strip()
    if missing(a) or missing(b): return result('MISSING_OR_PARTIAL_INTERVAL')
    if a=='1678/01/01' or b=='1678/01/01': return result('UGLC_NODATA_SENTINEL')
    try:
        if not all(re.fullmatch(r'\d{4}/\d{2}/\d{2}',x) for x in (a,b)): return result('FORMAT_REVIEW')
        x,y=[datetime.strptime(z,'%Y/%m/%d').date() for z in (a,b)]
        return window(x,y,'DAY' if x==y else 'SOURCE_INTERVAL',as_of)
    except ValueError: return result('INVALID_CALENDAR_DATE')

def glc(s,as_of):
    s=s.strip()
    if missing(s): return result('MISSING')
    for fmt in ('%m/%d/%Y %I:%M:%S %p','%m/%d/%Y','%Y-%m-%d'):
        try:
            x=datetime.strptime(s,fmt).date()
            return window(x,x,'DAY',as_of)
        except ValueError: pass
    return result('INVALID_OR_UNSUPPORTED_DATE')

def gsi(s,as_of):
    s=' '.join(s.strip().casefold().split())
    if missing(s): return result('MISSING')
    # Only standalone, unambiguous patterns. History is not necessarily occurrence.
    try:
        if re.fullmatch(r'\d{4}',s):
            y=int(s); return window(date(y,1,1),date(y,12,31),'YEAR',as_of)
        m=re.fullmatch(r'([a-z]+)\s+(\d{4})',s)
        if m:
            mon,y=m.groups()
            if mon not in MONTHS: return result('FORMAT_OR_MULTIPLE_REVIEW')
            y,mon=int(y),MONTHS[mon]
            return window(date(y,mon,1),date(y,mon,calendar.monthrange(y,mon)[1]),'MONTH',as_of)
        m=re.fullmatch(r'(\d{1,2})(?:st|nd|rd|th)?\s*[-–]\s*(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\s+(\d{4})',s)
        if m:
            a,b,mon,y=m.groups(); return window(dmy(y,MONTHS[mon],a),dmy(y,MONTHS[mon],b),'DAY_RANGE',as_of)
        m=re.fullmatch(r'(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\s+(\d{4})',s)
        if m:
            d,mon,y=m.groups(); x=dmy(y,MONTHS[mon],d); return window(x,x,'DAY',as_of)
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}',s):
            x=strict_iso(s); return window(x,x,'DAY',as_of)
        m=re.fullmatch(r'(\d{1,2})([-/])(\d{1,2})\2(\d{4})',s)
        if m:
            a,_,b,y=m.groups(); a,b,y=int(a),int(b),int(y)
            if 1<=a<=12 and 1<=b<=12 and a!=b: return result('AMBIGUOUS_NUMERIC_DAY')
            x=dmy(y,b,a) if a>12 else dmy(y,a,b)
            return window(x,x,'DAY',as_of)
        return result('FORMAT_OR_MULTIPLE_REVIEW')
    except (ValueError,KeyError): return result('INVALID_CALENDAR_DATE')

def coolr(s,as_of):
    s=s.strip(); extra={'encoded_timestamp_utc':'','alternative_ist_day':''}
    if missing(s): return result('MISSING')|extra
    # Source field verified to be ArcGIS date; only known 12/13 digit milliseconds.
    if not re.fullmatch(r'\d{12,13}(?:\.0+)?',s): return result('EPOCH_FORMAT_REVIEW')|extra
    try:
        ms=int(Decimal(s)); dt=datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(milliseconds=ms)
        utc,local=dt.date(),dt.astimezone(IST).date()
        extra={'encoded_timestamp_utc':dt.isoformat(),'alternative_ist_day':local.isoformat()}
        # UTC and IST dates are diagnostics, not proof of source event timezone.
        if utc!=local:
            return window(min(utc,local),max(utc,local),'UTC_IST_DAY_AMBIGUITY',as_of,'TIMEZONE_DAY_REVIEW')|extra
        return window(utc,utc,'DAY',as_of,'ENCODED_DAY_UNVERIFIED')|extra
    except (ValueError,OverflowError): return result('INVALID_EPOCH')|extra

def satellite(row, temporal):
    a,b=row['sat_date_before'],row['sat_date_after']; extra={'sat_before_raw':a,'sat_after_raw':b,'sat_before':'','sat_after':''}
    if missing(a) and missing(b): return extra|{'sat_window_status':'MISSING'}
    try:
        x=None if missing(a) else strict_iso(a.strip()); y=None if missing(b) else strict_iso(b.strip())
        extra.update(sat_before=iso(x),sat_after=iso(y))
        if x is None or y is None: status='PARTIAL'
        elif x>y: status='REVERSED'
        elif not temporal['start_date']: status='NO_EVENT_DATE_FOR_COMPARISON'
        elif strict_iso(temporal['end_date'])<x or strict_iso(temporal['start_date'])>y: status='EVENT_WINDOW_DISJOINT_REVIEW'
        elif strict_iso(temporal['start_date'])<x or strict_iso(temporal['end_date'])>y: status='EVENT_WINDOW_PARTLY_OUTSIDE_REVIEW'
        else: status='EVENT_WINDOW_WITHIN_SATELLITE_DATES'
    except ValueError: status='INVALID_SATELLITE_DATE'
    return extra|{'sat_window_status':status}

def gps(row,as_of):
    a,b=row['DateTimeS'],row['DateTime']; extra=dict(survey_timestamp_raw=a,survey_date_raw=b,survey_date='',survey_status='MISSING')
    values=[]; invalid=False; sentinel=False
    for s,kind in [(a,'timestamp'),(b,'date')]:
        s=s.strip()
        if missing(s): continue
        if s=='1899-12-30': sentinel=True; continue
        try:
            if kind=='timestamp':
                if not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z',s): raise ValueError()
                x=datetime.fromisoformat(s.replace('Z','+00:00')).date()
            else: x=strict_iso(s)
            if x>as_of: raise ValueError()
            values.append(x)
        except ValueError: invalid=True
    if invalid: status='SURVEY_FORMAT_OR_DATE_REVIEW'
    elif len(set(values))>1: status='SURVEY_DATE_CONFLICT'
    elif values: status='SURVEY_DAY_CANDIDATE_WITH_SENTINEL' if sentinel else 'SURVEY_DAY_CANDIDATE'; extra['survey_date']=iso(values[0])
    elif sentinel: status='PLACEHOLDER_DATE_ONLY'
    else: status='MISSING'
    extra['survey_status']=status
    return extra

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f); require(r.fieldnames and len(set(r.fieldnames))==len(r.fieldnames),f'Invalid header {path}')
        rows=list(r); require(all(None not in x and all(v is not None for v in x.values()) for x in rows),f'Malformed rows {path}')
        return rows

def indexed(path,n):
    rows=read(path); ids={x['qa_record_id'] for x in rows}
    require(len(rows)==len(ids)==n and '' not in ids,f'Count or ID mismatch {path}'); return rows

def links(path):
    pairs={(r['qa_record_id'],r.get('qa_source_layer','')) for r in read(path)}
    require(len({i for i,l in pairs})==len(pairs) and all(i for i,l in pairs),f'Conflicting source layers {path}')
    return pairs

def check_ids(rows,pairs):
    require({r['qa_record_id'] for r in rows}=={i for i,l in pairs},'Candidate vs district-link ID mismatch')

def write(path,rows,fields):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def process(base,as_of):
    output=[]
    def add(src,layer,id,role,t,**extra):
        output.append(dict(source=src,layer=layer,qa_record_id=id,date_role=role,representation_status='BASELINE_CANDIDATE',**t)|extra)
    p=base/'integration_preflight_v1'; require(json.loads((p/'complete.json').read_text(encoding='utf-8'))['status']=='PASS','Preflight incomplete')
    g=p/'glc_regional'; r=indexed(g/'glc_candidate_attributes.csv',1179); check_ids(r,links(g/'glc_district_links.csv'))
    for x in r: add('GLC','points',x['qa_record_id'],'REPORTED_EVENT_DATE',glc(x['event_date'],as_of),raw_start=x['event_date'],raw_time=x['event_time'])
    g=base/'gsi_quality_v1'; r=indexed(g/'gsi_point_history_audit.csv',27484); gp=links(base/'gsi_regional_v1/gsi_district_links.csv'); pt={(i,l) for i,l in gp if l.startswith('LANDSLIDE_POINT_')}; poly={(i,l) for i,l in gp if l.startswith('LANDSLIDE_POLYGON_')}
    require(len(gp)==111958 and len(poly)==84474 and len(pt)+len(poly)==len(gp),'GSI link inventory mismatch'); check_ids(r,pt)
    require({(x['qa_record_id'],x['source_layer']) for x in r}==pt,'GSI layer mismatch')
    for x in r: add('GSI',x['source_layer'],x['qa_record_id'],'HISTORY_UNVERIFIED',gsi(x['history_raw'],as_of),raw_start=x['history_raw'])
    for i,l in sorted(poly): add('GSI',l,i,'NO_DEDICATED_DATE',result('NO_DEDICATED_DATE'))
    for kind,n in [('points',2110),('polygons',5430)]:
        r=indexed(base/f'uglc_quality_v1/{kind}/date_accuracy_candidates.csv',n); check_ids(r,links(base/f'uglc_regional_v1/{kind}/uglc_district_links.csv'))
        for x in r: add('UGLC',kind,x['qa_record_id'],'SOURCE_REPORTED_WINDOW',uglc(x['START DATE'],x['END DATE'],as_of),raw_start=x['START DATE'],raw_end=x['END DATE'])
    g=base/'coolr_regional_v1'; r=indexed(g/'coolr_candidate_attributes.csv',1504); check_ids(r,links(g/'coolr_district_links.csv'))
    require(Counter(x['qa_source_layer'] for x in r)=={'COOLR_Events_Points':177,'COOLR_Reports_Points':1323,'COOLR_Reports_Polygons':4},'COOLR layer mismatch')
    for x in r:
        t=coolr(x['event_date'],as_of); add('COOLR',x['qa_source_layer'],x['qa_record_id'],'ENCODED_EVENT_DATE',t,raw_start=x['event_date'],raw_time=x['event_time'],**satellite(x,t))
    g=base/'lesser_himalayas_regional_v1'; r=indexed(g/'gps_original_attributes.csv',359); lp=links(g/'lesser_district_links.csv'); gps_links={(i,l) for i,l in lp if l=='Field_GPS_landslides'}; check_ids(r,gps_links)
    require(Counter(l for i,l in lp)=={'Field_GPS_landslides':359,'landslides_shimla':3153,'landslides_shimla_points':3153},'Lesser link count mismatch')
    ctx={'context_start':'2023-07-01','context_end':'2023-08-31'}
    for x in r: add('LESSER_HIMALAYAS_2023','Field_GPS_landslides',x['qa_record_id'],'SURVEY_NOT_OCCURRENCE',result('NO_INDIVIDUAL_OCCURRENCE_DATE'),**gps(x,as_of),**ctx)
    for i,l in sorted(lp-gps_links):
        add('LESSER_HIMALAYAS_2023',l,i,'INVENTORY_CONTEXT_ONLY',result('NO_INDIVIDUAL_OCCURRENCE_DATE'),**ctx,representation_status='ASSOCIATED_POINT_NOT_INDEPENDENT' if l.endswith('_points') else 'BASELINE_CANDIDATE')
    require(len(output)==128846,'Total candidate count mismatch')
    return output

def export(rows,out,as_of):
    keys={(x['source'],x['qa_record_id']) for x in rows}; require(len(keys)==len(rows),'Duplicate source record key')
    for x in rows:
        a,b=x['start_date'],x['end_date']; require(bool(a)==bool(b),'Partial normalized interval')
        if a: require(strict_iso(a)<=strict_iso(b)<=as_of and x['interval_days']==(strict_iso(b)-strict_iso(a)).days+1,'Invalid interval')
    write(out/'temporal_candidates.csv',rows,FIELDS)
    saved=read(out/'temporal_candidates.csv'); require(len(saved)==len(rows) and {(x['source'],x['qa_record_id']) for x in saved}==keys,'Readback mismatch')
    require(all(all(y[k]==str(x.get(k,'')) for k in FIELDS) for x,y in zip(rows,saved)), 'Readback attribute mismatch')
    del saved
    counts=Counter((x['source'],x['date_role'],x['date_status'],x['date_precision']) for x in rows)
    summary=[dict(source=s,date_role=r,date_status=t,date_precision=p,records=n) for (s,r,t,p),n in sorted(counts.items())]
    write(out/'temporal_status_counts.csv',summary,['source','date_role','date_status','date_precision','records'])
    daily=[x for x in rows if x['date_precision']=='DAY' and x['start_date'] and x['start_date']==x['end_date']]
    write(out/'day_candidates_unverified.csv',daily,FIELDS)
    survey=Counter(x.get('survey_status','') for x in rows if x['date_role']=='SURVEY_NOT_OCCURRENCE'); sat=Counter(x.get('sat_window_status','') for x in rows if x['source']=='COOLR')
    years=Counter((x['source'],x['start_date'][:4]) for x in daily)
    write(out/'day_candidate_year_counts.csv',[dict(source=s,year=y,records=n) for (s,y),n in sorted(years.items())],['source','year','records'])
    for x in summary: print(x['source'],x['date_status'],x['date_precision'],x['records'])
    print('Survey status:',dict(survey)); print('Satellite window status:',dict(sat)); print('Unverified day candidates by source:',dict(Counter(x['source'] for x in daily)))
    report=dict(status='PASS',as_of=iso(as_of),records=len(rows),source_counts=dict(Counter(x['source'] for x in rows)),day_candidates_unverified=len(daily),associated_points=sum(x['representation_status']=='ASSOCIATED_POINT_NOT_INDEPENDENT' for x in rows),survey_status=dict(survey),satellite_window_status=dict(sat),approved_training_labels=0,excluded_repaired_review_polygons=6)
    (out/'complete.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))

def self_test():
    a=date(2026,11,28)
    assert uglc('1678/01/01','2020/01/01',a)['date_status']=='UGLC_NODATA_SENTINEL'
    assert uglc('2024/02/01','2024/02/29',a)['interval_days']==29
    assert uglc('2024/02/29','2024/02/01',a)['date_status']=='REVERSED_INTERVAL'
    assert glc('09/19/2010 12:00:00 AM',a)['start_date']=='2010-09-19'
    assert coolr('1688860800000.0',a)['start_date']=='2023-07-09'
    ms=int(datetime(2023,7,9,22,tzinfo=timezone.utc).timestamp()*1000)
    assert coolr(str(ms),a)['date_status']=='TIMEZONE_DAY_REVIEW'
    assert gsi('16th-17th June 2013',a)['interval_days']==2
    assert gsi('February 2024',a)['interval_days']==29
    assert gsi('2015',a)['interval_days']==365
    assert gsi('02/03/2023',a)['date_status']=='AMBIGUOUS_NUMERIC_DAY'
    assert gsi('31 February 2024',a)['date_status']=='INVALID_CALENDAR_DATE'
    assert gsi('2017, 11 July 2018',a)['date_status']=='FORMAT_OR_MULTIPLE_REVIEW'
    assert gsi('2027',a)['date_status']=='FUTURE_DATE_REVIEW'
    assert gps({'DateTimeS':'2023-10-26T04:29:56Z','DateTime':'1899-12-30'},a)['survey_date']=='2023-10-26'
    assert gps({'DateTimeS':'','DateTime':'1899-12-30'},a)['survey_status']=='PLACEHOLDER_DATE_ONLY'
    assert satellite({'sat_date_before':'2023-06-16','sat_date_after':'2023-07-31'},coolr('1688860800000.0',a))['sat_window_status']=='EVENT_WINDOW_WITHIN_SATELLITE_DATES'
    with TemporaryDirectory() as t:
        row=dict(source='GLC',layer='points',qa_record_id='GLC:1',date_role='REPORTED_EVENT_DATE',representation_status='BASELINE_CANDIDATE',**glc('09/19/2010 12:00:00 AM',a))
        export([row],Path(t),a)
    print('Synthetic parser, intervals, epoch, survey, satellite and CSV readback tests: PASS')

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--self-test',action='store_true'); p.add_argument('--as-of',default=date.today().isoformat()); args=p.parse_args()
    if args.self_test: self_test(); return
    a=strict_iso(args.as_of); base=ROOT/'data/processed/landslides'; out=base/'temporal_standardization_v1'; stage=out.with_name(out.name+'.staging')
    require(not out.exists() and not stage.exists(),'Output or staging exists; inspect before retrying; no overwrite')
    rows=process(base,a); stage.mkdir(); export(rows,stage,a); stage.rename(out)
    print('Output:',out,'| Original data and prior processed outputs unchanged.')
if __name__=='__main__': main()
