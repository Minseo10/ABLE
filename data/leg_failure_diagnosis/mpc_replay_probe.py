import json,sys,time
from pathlib import Path
import numpy as np
import pybullet as p
from kinder_pddlstream_planning.limbrepositioning3d.run import create_env,build_stream_context,reset_to_start
from kinder_pddlstream_planning.limbrepositioning3d.stream import LimbConf,MPCConfig,PredictiveSamplingMPC,sample_grasp,plan_grasp_motion
from kinder_pddlstream_planning.limbrepositioning3d import utils as u

variant=sys.argv[1]
scale=float(sys.argv[2])
out=Path(sys.argv[3])
rollouts=int(sys.argv[4]) if len(sys.argv)>4 else 16
sim=create_env(variant,muscle_tone='none')
sim.reset(seed=0)
ctx=build_stream_context(sim,motion_seed=0,mpc=MPCConfig(num_rollouts=rollouts))
reset_to_start(ctx)
q0=tuple(sim.config.scene.limb_init_joint_positions)
goal=np.array(sim.config.scene.limb_goal_joint_positions)
grasp=next(sample_grasp(ctx,'limb'))[0]
gen=plan_grasp_motion(ctx,'limb',grasp,LimbConf(q0),LimbConf(tuple(goal)))
_,_,initial=next(gen);gen.close()
if scale!=1:
 for i in range(p.getNumJoints(sim.limb.robot_id,physicsClientId=sim.physics_client_id)):
  info=p.getDynamicsInfo(sim.limb.robot_id,i,physicsClientId=sim.physics_client_id)
  if info[0]>0:p.changeDynamics(sim.limb.robot_id,i,mass=info[0]*scale,localInertiaDiagonal=[v*scale for v in info[2]],physicsClientId=sim.physics_client_id)
u.restore_state(sim,initial)
mpc=PredictiveSamplingMPC(ctx,goal,initial)
commands=[];trace=[];started=time.monotonic();scores=[]
original_score=mpc._score
def score(*args):
 result=original_score(*args);scores.append(result);return result
mpc._score=score
for step in range(200):
 if u.limb_error(ctx,sim.limb.get_joint_positions(),goal)<.05:break
 scores.clear()
 action=mpc.step()[0]
 for _ in range(12):
  u.advance_corrected(sim,action)
  commands.append(list(action))
  trace.append(list(sim.limb.get_joint_positions()))
  if u.limb_error(ctx,sim.limb.get_joint_positions(),goal)<.05:break
 if step%20==19:print(variant,scale,step+1,'error',u.limb_error(ctx,sim.limb.get_joint_positions(),goal),'valid',sum(s[0]==0 for s in scores),flush=True)
 if u.limb_out_of_limits(ctx) or u.human_in_collision(ctx) or u.arm_in_collision(ctx):break
u.restore_state(sim,initial)
replay=[]
for action in commands:
 u.advance_corrected(sim,action)
 replay.append(list(sim.limb.get_joint_positions()))
delta=np.abs(np.asarray(trace)-np.asarray(replay))
result=dict(variant=variant,mass_scale=scale,rollouts=rollouts,control_steps=step+1,physics_steps=len(commands),wall_seconds=time.monotonic()-started,closed_error=u.limb_error(ctx,trace[-1],goal),replay_error=u.limb_error(ctx,replay[-1],goal),max_joint_trace_discrepancy=float(delta.max()),final_joint_trace_discrepancy=float(delta[-1].max()),trace=trace,replay=replay,commands=commands)
out.write_text(json.dumps(result,indent=2))
print('RESULT', {k:v for k,v in result.items() if k not in ('trace','replay','commands')},flush=True)
sim.close()
