import os,subprocess,json,datetime
root='/diskA/YYD/TimeRCD'
exp='/diskA/YYD/TimeRCD/experiments/ablation_v9'
items=[('ignore', 'p0_ignore_dp_20261112_170107', 'p0_ignore_dp_20261112_170107_eval_20261114_restart2_s200_r700'), ('official', 'p0_official_dp_20261111_003608', 'p0_official_dp_20261111_003608_eval_20261114_restart2_s200_r700')]
py='/home/omnisky/anaconda3/envs/time-rcd/bin/python3'
entry=os.path.join(exp,'run_experiment.py')
env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']='0,1'; env['OMP_NUM_THREADS']='1'; env['MKL_NUM_THREADS']='1'
summary={'scope':'fixed synthetic held-out first 200 + all 700 real','started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':[]}
for label,train_run,eval_run in items:
 cfg='experiments/ablation_v9/runs/'+train_run+'/config.json'; ck='experiments/ablation_v9/runs/'+train_run+'/checkpoints/best.pth'
 cmd=[py,entry,'--config',cfg,'--checkpoint',ck,'--eval-config','experiments/ablation_v9/configs/eval_official.json','--run-id',eval_run,'--parallel','data_parallel','--gpu-ids','0','1','--eval-max-synthetic','200','--mode','eval']
 rec={'model':label,'run_id':eval_run,'command':cmd,'started':datetime.datetime.now(datetime.timezone.utc).isoformat()}; summary['runs'].append(rec)
 with open(os.path.join(exp,'audits/eval_20261114/restart2_eval_queue.json'),'w') as out: json.dump(summary,out,indent=2)
 p=subprocess.run(cmd,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 with open(os.path.join(exp,'audits/eval_20261114',eval_run+'_console.log'),'w') as out: out.write(p.stdout)
 rec['exit_code']=p.returncode; rec['completed']=datetime.datetime.now(datetime.timezone.utc).isoformat(); rec['status']='completed' if p.returncode==0 else 'failed'
 with open(os.path.join(exp,'audits/eval_20261114/restart2_eval_queue.json'),'w') as out: json.dump(summary,out,indent=2)
 if p.returncode: break
summary['finished']=datetime.datetime.now(datetime.timezone.utc).isoformat()
with open(os.path.join(exp,'audits/eval_20261114/restart2_eval_queue.json'),'w') as out: json.dump(summary,out,indent=2)
