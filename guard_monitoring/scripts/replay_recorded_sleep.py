"""Replay recorded sensor decisions, not video inference. Run from project root:
python -m guard_monitoring.scripts.replay_recorded_sleep --output replay.json
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from guard_monitoring.diagnostics import BUILD_ID
from guard_monitoring.monitoring.activity import ActivityStabilizer
from guard_monitoring.monitoring.rules import TimedRuleEngine
from guard_monitoring.monitoring.sleep import SleepAnalyzer
from guard_monitoring.types import EyeState, MovementState, PhoneState, PostureState


def replay():
    path=Path(__file__).resolve().parents[1]/'tests/fixtures/sleep_eye_evidence.json'
    fixture=json.loads(path.read_text(encoding='utf-8'))
    analyzer=SleepAnalyzer({});activity=ActivityStabilizer({})
    # Explicit replay settings; never read/modify the deployment's environment.
    rule_cfg=dict(warmup_seconds=0,sleep_seconds=5,phone_seconds=5,
                  stationary_seconds=8,absence_seconds=5,
                  grace_seconds=dict(sleep=10,phone=3,stationary=3,absence=1),
                  max_evidence_gap_seconds=2,unknown_reset_seconds=5)
    rules=TimedRuleEngine('recorded-sensor-replay',rule_cfg)
    decisions=[];events=[]
    for r in fixture['frames']:
        t=r['time_seconds']
        state=analyzer.update(r['guard_track_id'],t,EyeState(**r['eyes']),
                             PostureState(**r['posture']),MovementState(),PhoneState(**r['phone']))
        condition=True if state.candidate else False if state.evidence_quality=='high' else None
        event=rules.update('sleep',condition,t,r['guard_track_id'],{})
        if event:events.append(event.to_dict())
        shown=activity.update('Sleeping' if state.candidate else None,t,present=True)
        decisions.append(dict(time_seconds=t,candidate=state.candidate,label=shown['label'],
                              reason=state.reason,basis=state.decision_basis,
                              eye_seconds=state.eye_evidence_seconds,perclos=state.perclos,
                              sleep_timer_seconds=rules.active_duration('sleep',t)))
    candidates=[r for r in decisions if r['candidate']]
    shown=[r for r in decisions if r['label']=='Sleeping']
    return dict(build_id=BUILD_ID,scope=fixture['description'],
                assumptions=['Motion is unknown because original logs have no box/camera validation.',
                             'Original eye, posture and phone signals reused; no new model inference.',
                             'No scene cuts simulated: original logs do not contain cut evidence.',
                             'Analyzer/presentation use code defaults independent of deployment .env.',
                             'Five-second event threshold is an explicit replay setting.'],
                frames=len(decisions),candidate_frames=len(candidates),sleep_display_frames=len(shown),
                first_candidate_seconds=candidates[0]['time_seconds'] if candidates else None,
                first_sleep_display_seconds=shown[0]['time_seconds'] if shown else None,
                max_sleep_timer_seconds=max(r['sleep_timer_seconds'] for r in decisions),
                sleep_events=events,replay_rule_config=rule_cfg,decisions=decisions)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();result=replay()
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='decisions'},indent=2))
