# Kitchen PPE validation

The pipeline has no source filename, STAFF number or frame-number rules. The same
inference, ownership, confirmation and identity rules apply to every session.
This does not establish that the trained models are accurate in every scene.

## Behavior and configuration

- All four requirements can receive a second inference view when unresolved.
  Current full-frame evidence is not replaced just to obtain a preferred label.
- Small people may be retried immediately. Nearby people become eligible after
  a sustained unresolved interval. These policies, inference sizes and the
  per-frame target budget are in config.py; they are starting defaults, not
  thresholds proven on every camera.
- Recheck scheduling uses stable identity-based fairness. An inference budget
  of two people is not a limit on tracked people or displayed cards.
- Crop boxes are mapped back to source coordinates. Ambiguous ownership remains
  unassigned. A crop containing a person does not prove that every detection in
  it belongs to that person.
- Three distinct observations are required. The vote window follows observed
  processing cadence, bounded between one and six seconds. Empty frames cannot
  vote. Multiple inference views of one source frame cannot add extra votes.
- Confirmed statuses expire after their configured hold times. A contradictory
  class clears the held status immediately. Longer retention is not evidence of
  a new detection; the API records evidence_timestamp and evidence_age_seconds.
- Slow continuous observations do not expire an otherwise continuous native
  track. An actual missing-person gap does expire its evidence and admission.
- A long camera disconnection starts a new tracking generation. IDs and PPE
  state are not silently transferred across a period with no observations.
  Short occlusions use the tracker buffer. Long-term re-identification and
  cross-camera identity are not implemented or guaranteed.
- The display stays minimal: person portrait, ID and four fixed rows. No hidden
  label or explanatory text is added to the video.

## Automated checks

Run from the repository root:

    python -m unittest discover -s kitchen_monitoring/tests -v

Tests cover confirmation at 0.5, 1, 5, 15, 30 and 60 observations per second;
slow processing; crowd scheduling and reordered detections; overlap/crop
ownership; new entrants; temporary loss and reacquisition; long disconnection;
exact negative classes; contradictory evidence; duplicate votes; and expiry.
These are logic tests, not measurements of detector precision or recall.

## Repeatable real-video smoke tests

Create a JSON manifest; video paths are relative to the manifest:

    [
      {"name": "camera_day", "video": "day.mp4", "start_frame": 0, "frames": 90},
      {"name": "camera_overlap", "video": "overlap.mp4", "start_frame": 300, "frames": 90}
    ]

Run in a separate process:

    python -m kitchen_monitoring.validation --manifest cases.json --output review_results

The runner records model results, observed track counts, per-class coverage,
processing time, screenshots and annotated video. It uses isolated output and
mocked session/violation storage. It does not modify production session records.
An incomplete requested segment returns a nonzero exit code. The accuracy field
is deliberately null without reference labels; fewer IDs or fewer blank rows do
not themselves establish correctness.

## Required before a broad accuracy claim

Maintain separate tuning and held-out videos from actual deployed cameras.
Label each visible person, identity continuity, visible PPE regions and exact
classes. Include explicit no-apron/no-glove/no-mask/incorrect-mask examples,
not only compliant workers. Include close/distant workers, partial bodies,
crossings, long exits, crowds, day/night lighting, different uniforms, camera
angles, motion blur and camera outages. Mark unobservable items separately.

For each scene, measure person detection misses and false positives, ID switches,
wrong-person PPE assignment, per-class precision/recall, unknown coverage,
status-change latency and processing latency. Report results per scene/class,
not just an overall percentage. Set business acceptance targets before tuning.
If raw predictions remain wrong across supported views, dataset/model improvement
is required; pipeline logic cannot safely invent a missing result.
