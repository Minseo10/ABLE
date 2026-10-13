import json
from pathlib import Path
import time
import numpy as np
import pybullet as p
from kinder_pddlstream_planning.limbrepositioning3d.run import create_env, build_stream_context, reset_to_start
from kinder_pddlstream_planning.limbrepositioning3d.stream import LimbConf, sample_grasp, plan_grasp_motion
from kinder_pddlstream_planning.limbrepositioning3d import utils as u

out=Path('/home/minseo/research/learning-forceful-manipulation/data/leg_failure_diagnosis')
out.mkdir(exist_ok=True)
rows=[]
for scene in ('isolated','human','bed','wheelchair'):
 for limb in ('left-arm','right-arm','left-leg','right-leg'):
  variant=f'{scene}-{limb}'
  sim=create_env(variant, muscle_tone='none')
  sim.reset(seed=0)
  ctx=build_stream_context(sim,motion_seed=0)
  reset_to_start(ctx)
  init=sim.scene.config.limb_init_joint_positions if hasattr(sim.scene,'config') else sim.config.scene.limb_init_joint_positions
  goal=sim.config.scene.limb_goal_joint_positions
  grasp=next(sample_grasp(ctx,'limb'))[0]
  gen=plan_grasp_motion(ctx,'limb',grasp,LimbConf(tuple(init)),LimbConf(tuple(goal)))
  _,_,state=next(gen)
  gen.close()
  u.restore_state(sim,state)
  mass=sum(p.getDynamicsInfo(sim.limb.robot_id,i,physicsClientId=sim.physics_client_id)[0] for i in range(-1,p.getNumJoints(sim.limb.robot_id,physicsClientId=sim.physics_client_id)))
  metrics=[]
  seed=state.robot_positions
  for q in np.linspace(init,goal,11):
   sol=u.grasp_ik(ctx,grasp,tuple(q),seed)
   if sol is None:
    metrics.append({'ik':False});continue
   seed=list(sol)
   sim.robot.arm.set_joints(seed)
   jl=u._tool_jacobian(sim,sim.limb)
   jr=u._tool_jacobian(sim,sim.robot.arm)[:,:7]
   mm=np.asarray(p.calculateMassMatrix(sim.limb.robot_id,list(q),physicsClientId=sim.physics_client_id))
   metrics.append(dict(ik=True,limb_sigma_min=float(np.linalg.svd(jl,compute_uv=False)[-1]),robot_sigma_min=float(np.linalg.svd(jr,compute_uv=False)[-1]),hold_ratio=u.hold_saturation(sim),limb_mass_matrix_trace=float(np.trace(mm)),limb_gravity_max=float(np.max(abs(u._gravity_torque(sim,sim.limb))))))
  row=dict(variant=variant,moving_limb_mass_kg=mass,limb_joints=len(sim.limb.arm_joints),init=list(init),goal=list(goal),goal_distance=u.limb_error(ctx,list(init),np.array(goal)),path=metrics)
  if scene=='isolated':
   traces={}
   for mode in ('no_scratch','joint_restore','bullet_restore'):
    u.restore_state(sim,state)
    trace=[]
    rng=np.random.default_rng(13)
    for control in range(30):
     if mode!='no_scratch':
      snapshot=u.capture_state(sim)
      sid=p.saveState(physicsClientId=sim.physics_client_id) if mode=='bullet_restore' else None
      for candidate in range(4):
       if sid is None:u.restore_state(sim,snapshot,regrasp=False)
       else:p.restoreState(stateId=sid,physicsClientId=sim.physics_client_id)
       correction=rng.normal(0,.12,7)
       for _ in range(144):u.advance_corrected(sim,correction)
      if sid is None:u.restore_state(sim,snapshot,regrasp=False)
      else:
       p.restoreState(stateId=sid,physicsClientId=sim.physics_client_id)
       p.removeState(sid,physicsClientId=sim.physics_client_id)
     for _ in range(12):u.advance_corrected(sim,[.1]*7)
     trace.append(dict(limb_q=list(sim.limb.get_joint_positions()),robot_q=list(sim.robot.arm.get_joint_positions())))
    traces[mode]=trace
   row['scratch_test']=traces
  rows.append(row)
  (out/'comparison.json').write_text(json.dumps(rows,indent=2))
  print(variant,'mass',round(mass,3),'sigma_min',round(min(r['limb_sigma_min'] for r in metrics if r['ik']),5),'max_hold_ratio',round(max(r['hold_ratio'] for r in metrics if r['ik']),3),flush=True)
  sim.close()
