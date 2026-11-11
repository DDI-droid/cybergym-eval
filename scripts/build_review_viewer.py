#!/usr/bin/env python3
"""Render a private, dependency-free viewer from existing baseline evidence only.
No models, grading, or network requests. Original run artifacts are never changed.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import shlex
import sqlite3
from datetime import datetime, timezone

LABELS = {
    'upstream_error': 'Upstream/API error',
    'no_final_tested': 'Tested, no final file',
    'no_final_untested': 'No server record, no final file',
    'final_no_crash': 'Final did not crash vulnerable build',
    'fixed_failed': 'Fixed build also failed',
    'passed': 'Passed',
    'other': 'Other / inspect evidence',
}

def redact(value):
    if isinstance(value, dict):
        return {k: '[REDACTED]' if re.search(r'api[_-]?key|authorization|password|secret|access_token', k, re.I) else redact(v) for k,v in value.items()}
    if isinstance(value, list): return [redact(x) for x in value]
    if not isinstance(value, str): return value
    value = re.sub(r'ABSK[A-Za-z0-9+/=]+|\bcge-[a-f0-9]{32}\b|\bsk-[A-Za-z0-9_-]{20,}', '[REDACTED]', value)
    value = re.sub(r'(?i)(Bearer\s+)[A-Za-z0-9._+/-]{16,}', r'\1[REDACTED]', value)
    return value

def command_of(step):
    if step.get('type') != 'function_call': return None
    arg = step.get('arguments', {})
    try:
        if isinstance(arg, str): arg = json.loads(arg)
    except ValueError: return str(arg)
    command = arg.get('command', arg.get('cmd')) if isinstance(arg, dict) else arg
    if isinstance(command, str):
        try:
            parsed = json.loads(command)
            if isinstance(parsed, list): command = parsed
        except ValueError: pass
    return shlex.join([str(x) for x in command]) if isinstance(command, list) else str(command or '')

def classify(result, stderr, records):
    if result.get('passed'): return 'passed'
    if 'internal_server_error' in stderr or 'type: \'server_error\'' in stderr: return 'upstream_error'
    final = result.get('final')
    if not final: return 'no_final_tested' if records else 'no_final_untested'
    if final.get('vul_exit_code') in (0, 300): return 'final_no_crash'
    if final.get('fix_exit_code') is not None and final.get('fix_exit_code') != 0: return 'fixed_failed'
    return 'other'

def load_evidence(work, label):
    run = work / 'train25' / label
    paths = sorted(run.glob('*/result.json'))
    if not paths: raise ValueError(f'No saved results under {run}')
    db = sqlite3.connect((work / 'server/poc.db').resolve().as_uri()+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    tasks = []
    try:
        for p in paths:
            r = json.loads(p.read_text())
            if r['label'] != label: raise ValueError(f'Label mismatch in {p}')
            folder = p.parent
            trajectory = folder / 'output/trajectory.jsonl'
            steps = []
            if trajectory.exists():
                for n, line in enumerate(trajectory.read_text(errors='replace').splitlines(), 1):
                    if line.strip():
                        try: steps.append(json.loads(line))
                        except ValueError: steps.append({'type':'unparsed', 'line':n, 'text':line})
            stderr_path = folder / 'output/codex.err'
            stderr = stderr_path.read_text(errors='replace') if stderr_path.exists() else ''
            records = [dict(x) for x in db.execute('SELECT poc_id,poc_hash,poc_length,vul_exit_code,fix_exit_code,created_at FROM poc_records WHERE agent_id=? ORDER BY created_at,poc_id', (r['agent_id'],))]
            commands = [(i, command_of(s)) for i,s in enumerate(steps)]
            commands = [(i,c) for i,c in commands if c is not None]
            category = classify(r, stderr, records)
            final = r.get('final') or {}
            final_record = next((x for x in records if x['poc_id'] == final.get('poc_id')), None)
            saved = folder / 'output/poc'
            digest = hashlib.sha256(saved.read_bytes()).hexdigest() if saved.is_file() else None
            matched = bool(final_record and digest and digest == final_record['poc_hash'])
            facts = [f"{len(commands)} recorded tool calls; {r.get('usage',{}).get('requests',0)} metered model requests.",
                     f"{len(records)} server database records (includes evaluator final submission when present).",
                     f"Agent exit {r.get('agent',{}).get('exit_code')}; {r.get('wall_seconds',0):.1f} seconds total."]
            if commands:
                facts += ['First command: '+commands[0][1], 'Last command: '+commands[-1][1]]
            if final: facts.append(f"Final grading: vulnerable={final.get('vul_exit_code')}, fixed={final.get('fix_exit_code')}; saved-file hash match={matched}.")
            else: facts.append('No final output was graded. This alone does not prove the agent never tested.')
            interpretations = {
                'passed': 'The designated final file passed vulnerable/fixed grading. Preserve this task as a positive control; success here does not establish generalization.',
                'upstream_error': 'The recorded stderr shows an upstream stream error and the agent stopped. Treat this as a transport/recovery failure, not a clean measure of model capability. Preserve the original score.',
                'no_final_tested': 'The server received test submissions, but no final file was graded. Inspect the actual test outputs and stopping behavior. Saving an already-unsuccessful candidate alone would not demonstrate a pass.',
                'no_final_untested': 'No server record exists for this agent and no final file was graded. Inspect command failures and stopping behavior before claiming it voluntarily gave up or reached a capability ceiling.',
                'final_no_crash': 'A final candidate was graded but did not trigger the required vulnerable-build crash. This is an observed miss under this harness, not proof that prompt/tool improvements cannot help.',
                'fixed_failed': 'The final candidate also fails the fixed-build requirement. A vulnerable-build nonzero exit by itself is not sufficient for success.',
                'other': 'The saved status does not fit a verified category. Review raw result and server records before assigning a cause.',
            }
            tasks.append({'task':r['task'], 'category':category, 'result':r, 'records':records,
                          'facts':facts, 'interpretation':interpretations[category], 'stderr':stderr,
                          'steps':steps, 'commands':len(commands), 'source':str(p.relative_to(work)),
                          'trajectory_present':trajectory.exists(), 'hash_match':matched,
                          'trajectory_sha256':hashlib.sha256(trajectory.read_bytes()).hexdigest() if trajectory.exists() else None})
    finally: db.close()
    return redact({'label':label, 'historical_label':'codex-v2' if label=='codex' else label,
                   'generated':datetime.now(timezone.utc).isoformat(timespec='seconds'),
                   'categories':LABELS, 'tasks':tasks})

PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; img-src data:; base-uri 'none'; form-action 'none'">
<title>CyberGym · baseline trajectory review</title>
<style>
:root{color-scheme:dark;--bg:#0b1018;--panel:#131c29;--line:#2b384b;--muted:#a6b6cc;--text:#eff5ff;--accent:#86b9ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}header{padding:26px 28px 16px;border-bottom:1px solid var(--line)}h1{font-size:27px;letter-spacing:-.7px;margin:0}h2{font-size:20px;margin:0 0 12px}h3{font-size:16px}p{margin:8px 0}small,.muted{color:var(--muted)}.metrics{display:flex;gap:14px;flex-wrap:wrap;margin-top:18px}.metric{background:var(--panel);padding:12px 18px;border:1px solid var(--line);border-radius:10px;min-width:135px}.metric strong{display:block;font-size:25px}.layout{display:grid;grid-template-columns:330px minmax(0,1fr);max-width:1600px;margin:auto}aside{padding:18px;border-right:1px solid var(--line);max-height:calc(100vh - 230px);overflow:auto;position:sticky;top:0}main{padding:24px;min-width:0}input,select,button{font:inherit;color:var(--text);background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:8px}input,select{width:100%;margin:4px 0}button{cursor:pointer}button:hover,button:focus-visible{border-color:var(--accent)}button:disabled{opacity:.4;cursor:default}.task{display:block;width:100%;text-align:left;margin-top:8px;padding:11px}.task.selected{border-color:var(--accent);background:#1b304b}.task small{display:block}.badge{display:inline-block;background:#273952;border-radius:6px;padding:3px 8px;font-size:12px}.passed{background:#154f3d}.upstream_error{background:#6c3036}.panel{border:1px solid var(--line);background:var(--panel);border-radius:10px;padding:16px;margin:14px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.5 ui-monospace,monospace;margin:8px 0;max-height:550px;overflow:auto}summary{cursor:pointer;padding:8px 0}.row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.row input{flex:1;min-width:200px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;text-align:left;border-bottom:1px solid var(--line)}.bar{height:9px;border-radius:5px;background:var(--accent);margin:4px 0 10px}.step{border-bottom:1px solid var(--line);padding:6px 0}.step summary{font-size:13px}.warning{border-left:3px solid #efb969;padding-left:12px}.fact{font-size:14px;overflow-wrap:anywhere}a{color:var(--accent)}@media(max-width:800px){.layout{grid-template-columns:1fr}aside{position:static;max-height:320px;border-right:0;border-bottom:1px solid var(--line)}main,header{padding:16px}.metric{min-width:105px;padding:8px 12px}.metric strong{font-size:20px}}
</style></head><body><header><h1>CyberGym / baseline review</h1><p id="subtitle" class="muted"></p><div id="metrics" class="metrics"></div></header>
<div class="layout"><aside><label for="search">Find a task</label><input id="search" placeholder="Task ID or outcome"><label for="filter">Outcome</label><select id="filter"><option value="all">All outcomes</option></select><div id="distribution"></div><p id="shown" class="muted" aria-live="polite"></p><nav id="tasks" aria-label="Tasks"></nav></aside><main id="detail"><noscript>Enable JavaScript to inspect the embedded trajectories. No network access is required.</noscript></main></div>
<script id="evidence" type="application/json">__DATA__</script><script>
'use strict';
const D=JSON.parse(document.getElementById('evidence').textContent), $=id=>document.getElementById(id);
const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
const fmt=n=>Number(n||0).toLocaleString(undefined,{maximumFractionDigits:2});
const j=x=>JSON.stringify(x,null,2);const count={};D.tasks.forEach(t=>count[t.category]=(count[t.category]||0)+1);
$('subtitle').textContent=`train25 · ${D.label} (originally ${D.historical_label}; fixed Codex harness v2) · DeepSeek V3.2 · ${D.generated}`;
const passed=D.tasks.filter(t=>t.result.passed).length, cost=D.tasks.reduce((s,t)=>s+(t.result.usage?.cost_usd||0),0), totalSteps=D.tasks.reduce((s,t)=>s+t.steps.length,0);
for(const [v,label] of [[`${passed}/${D.tasks.length}`,'passed · original grading'],[`${Math.round(passed/D.tasks.length*100)}%`,'observed pass rate'],[`$${cost.toFixed(2)}`,'recorded model cost'],[count.upstream_error||0,'upstream-error exits'],[fmt(totalSteps),'saved trajectory events']]){const n=el('div',undefined,'metric');n.append(el('strong',v),el('small',label));$('metrics').append(n);}
for(const [key,label] of Object.entries(D.categories)){if(!count[key])continue;const option=el('option',`${label} (${count[key]})`);option.value=key;$('filter').append(option);const box=el('div');box.append(el('small',`${label} · ${count[key]}`));const bar=el('div',undefined,'bar');bar.style.width=`${count[key]/D.tasks.length*100}%`;box.append(bar);$('distribution').append(box);}
let selected=null;function tasks(){const q=$('search').value.toLowerCase(),filter=$('filter').value;const list=D.tasks.filter(t=>(filter==='all'||t.category===filter)&&`${t.task} ${D.categories[t.category]}`.toLowerCase().includes(q));$('tasks').replaceChildren();$('shown').textContent=`${list.length} of ${D.tasks.length} tasks`;for(const t of list){const b=el('button',undefined,'task'+(t.task===selected?' selected':''));b.append(el('b',t.task),el('small',D.categories[t.category]),el('small',`${fmt(t.result.wall_seconds)} s · $${Number(t.result.usage?.cost_usd||0).toFixed(2)} · ${t.records.length} server records`));b.onclick=()=>show(t);$('tasks').append(b);}}
function disclosure(title,text){const d=el('details');d.append(el('summary',title),el('pre',text));return d;}
function show(t){selected=t.task;tasks();const main=$('detail');main.replaceChildren();main.append(el('h2',t.task),el('span',D.categories[t.category],`badge ${t.category}`));main.append(el('p',`Saved grade: ${t.result.status} · agent exit ${t.result.agent?.exit_code} · timeout ${Boolean(t.result.agent?.timed_out)} · OOM ${Boolean(t.result.agent?.oom_killed)}`,'muted'));
const facts=el('section',undefined,'panel');facts.append(el('h3','What the evidence says'));for(const f of t.facts)facts.append(el('p',f,'fact'));main.append(facts);
const review=el('section',undefined,'panel');review.append(el('h3','Review / next investigation'),el('p',t.interpretation),el('p','Interpretation, not a proven capability limit. No new grading or model calls were made for this report.','muted'));main.append(review);
const submissions=el('details',undefined,'panel');submissions.append(el('summary',`Server evidence · ${t.records.length} records`));const scroll=el('div',undefined,'scroll'),table=el('table'),head=el('tr');for(const h of ['PoC ID','Bytes','Vulnerable exit','Fixed exit','Final?'])head.append(el('th',h));table.append(head);for(const r of t.records){const row=el('tr');for(const v of [r.poc_id,r.poc_length,r.vul_exit_code??'not recorded',r.fix_exit_code??'not evaluated',r.poc_id===t.result.final?.poc_id?'designated final':''])row.append(el('td',v));table.append(row);}scroll.append(table);submissions.append(scroll,el('p','Database records are authoritative for received PoCs; mentioning submit.sh in a command is not proof of a successful submission.','muted'));main.append(submissions);
if(t.stderr)main.append(disclosure('Agent stderr (full saved text)',t.stderr));main.append(disclosure('Raw result.json (credential-pattern redacted)',j(t.result)));
const section=el('section',undefined,'panel');section.append(el('h3',`Full saved trajectory · ${t.steps.length} events`));section.append(el('p','All saved events are embedded, not a first/last-step sample. Expand any event to see its entire content. Only credential-pattern redaction is applied. Saved traces may themselves omit upstream events.','muted'));if(!t.trajectory_present)section.append(el('p','Trajectory file is missing.','warning'));
const controls=el('div',undefined,'row'),input=el('input');input.placeholder='Search all event content';input.setAttribute('aria-label','Search trajectory');const type=el('select');type.style.width='auto';for(const key of ['all',...new Set(t.steps.map(s=>s.type||'unknown'))]){const o=el('option',key==='all'?'All event types':key);o.value=key;type.append(o);}controls.append(input,type);section.append(controls);const events=el('div'),pager=el('div',undefined,'row'),prev=el('button','Previous'),next=el('button','Next'),status=el('span',undefined,'muted');pager.append(prev,status,next);section.append(pager,events);main.append(section);let page=0;const texts=t.steps.map(j);function render(){const q=input.value.toLowerCase(),indexes=t.steps.map((s,i)=>i).filter(i=>(type.value==='all'||(t.steps[i].type||'unknown')===type.value)&&texts[i].toLowerCase().includes(q));page=Math.max(0,Math.min(page,Math.ceil(indexes.length/25)-1));events.replaceChildren();for(const i of indexes.slice(page*25,(page+1)*25)){const s=t.steps[i],d=el('details',undefined,'step');const preview=s.arguments||s.content||s.output||s.text||'';const text=typeof preview==='string'?preview:JSON.stringify(preview);d.append(el('summary',`#${i+1} · ${s.type||'unknown'} ${s.role||s.name||''} · ${text.replace(/\s+/g,' ').slice(0,140)}`));d.addEventListener('toggle',()=>{if(d.open&&!d.querySelector('pre'))d.append(el('pre',texts[i]));});events.append(d);}status.textContent=`${indexes.length} matches · page ${indexes.length?page+1:0}/${Math.ceil(indexes.length/25)}`;prev.disabled=page===0;next.disabled=(page+1)*25>=indexes.length;}prev.onclick=()=>{page--;render();};next.onclick=()=>{page++;render();};input.oninput=()=>{page=0;render();};type.onchange=input.oninput;render();
main.append(el('p',`Source: ${t.source} · trajectory SHA-256: ${t.trajectory_sha256||'missing'}`,'muted'));}
$('search').oninput=tasks;$('filter').onchange=tasks;tasks();show(D.tasks.find(t=>t.category==='upstream_error')||D.tasks[0]);
</script></body></html>'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--label',default='codex')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=load_evidence(args.work,args.label)
    encoded=json.dumps(data,ensure_ascii=True,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    page=PAGE.replace('__DATA__',encoded)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # Separate viewer only: explicitly refuse accidental overwrite of an artifact.
    with args.output.open('x',encoding='utf8') as f:f.write(page)
    print(json.dumps({'file':str(args.output),'bytes':args.output.stat().st_size,'tasks':len(data['tasks']),
                      'events':sum(len(t['steps']) for t in data['tasks']),
                      'categories':dict(collections.Counter(t['category'] for t in data['tasks'])),
                      'passing_final_hashes_verified':sum(t['hash_match'] for t in data['tasks'] if t['result'].get('passed'))},indent=2))
if __name__=='__main__':main()
