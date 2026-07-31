#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC="$ROOT/src/gnn_marl_training_DGTA_nobuffer"
RUNNER="$ROOT/scripts/run_cross_map_paper_eval.sh"
VALIDATOR="$ROOT/scripts/validate_formal_episode_artifacts.py"
PYTHON="${PYTHON:-python3}"
CKPT="${CKPT:?CKPT is required}"
TRAINING_SEED="${TRAINING_SEED:?TRAINING_SEED is required}"
METHOD_LABEL="${METHOD_LABEL:?METHOD_LABEL is required}"
SCENARIO_BANK="${SCENARIO_BANK:-paper_micro_v1}"
NUM_EPISODES="${NUM_EPISODES:-50}"
EVAL_SEED="${EVAL_SEED:-20260715}"
OUT_DIR="${OUT_DIR:-$ROOT/artifacts/eval/formal_micro_${METHOD_LABEL}_seed${TRAINING_SEED}}"
ROS_DOMAIN_ID_EVAL="${ROS_DOMAIN_ID_EVAL:-221}"
GAZEBO_PORT_EVAL="${GAZEBO_PORT_EVAL:-12221}"
CONTROLLER="${CONTROLLER:-policy}"
SAFETY_FILTER_ENABLE="${SAFETY_FILTER_ENABLE:-1}"
COLLECT_CF_TRACE="${COLLECT_CF_TRACE:-1}"
EXPERIMENT_CONDITION="${EXPERIMENT_CONDITION:-fixed_scenario}"

mkdir -p "$OUT_DIR/cases"
MANIFEST="$OUT_DIR/scenario_bank.json"

# Active legacy P0 workers use 231..233; map them to valid, disjoint blocks.
DOMAIN_BASE="$ROS_DOMAIN_ID_EVAL"
if (( DOMAIN_BASE >= 231 && DOMAIN_BASE <= 233 )); then
  DOMAIN_BASE=$((181 + 4 * (DOMAIN_BASE - 231)))
fi
PORT_BASE="$GAZEBO_PORT_EVAL"
if (( PORT_BASE >= 12431 && PORT_BASE <= 12433 )); then
  PORT_BASE=$((12431 + 4 * (PORT_BASE - 12431)))
fi
PYTHONPATH="$SRC" "$PYTHON" - <<PY
from gnn_marl_training.formal_scenario_bank import write_scenario_manifest
write_scenario_manifest("$MANIFEST", "$SCENARIO_BANK")
PY

mapfile -t CASES < <(
  PYTHONPATH="$SRC" "$PYTHON" - <<PY
from gnn_marl_training.formal_scenario_bank import get_scenario_bank
for case in get_scenario_bank("$SCENARIO_BANK"):
    print("|".join(map(str, (
        case.case_id, case.category, case.map_number, case.num_agents,
        case.max_episode_steps, case.num_dynamic_obstacles, case.obstacle_speed_mps,
    ))))
PY
)

case_complete() {
  local out="$1" map="$2" max_steps="$3"
  [[ -s "$out/episodes.csv" && -s "$out/episodes.jsonl" ]] || return 1
  "$PYTHON" "$VALIDATOR" \
    --csv "$out/episodes.csv" --jsonl "$out/episodes.jsonl" \
    --scenario-manifest "$MANIFEST" --expected-records "$NUM_EPISODES" \
    --map "$map:$NUM_EPISODES:$max_steps" >/dev/null
}

case_index=0
for entry in "${CASES[@]}"; do
  IFS='|' read -r case_id category map agents max_steps obstacles speed <<<"$entry"
  case_domain=$((DOMAIN_BASE + case_index))
  case_port=$((PORT_BASE + case_index))
  (( case_domain <= 232 )) || {
    echo "[FATAL] ROS domain $case_domain exceeds Fast DDS maximum 232" >&2
    exit 2
  }
  case_out="$OUT_DIR/cases/$case_id"
  mkdir -p "$case_out"
  echo "[scenario] case=$case_id category=$category map=$map agents=$agents domain=$case_domain port=$case_port"
  if case_complete "$case_out" "$map" "$max_steps"; then
    echo "[skip] validated case=$case_id records=$NUM_EPISODES"
    case_index=$((case_index + 1))
    continue
  fi
  CKPT="$CKPT" MAPS="$map" NUM_EPISODES="$NUM_EPISODES" NUM_AGENTS="$agents" \
    MAP3_NUM_DYNAMIC_OBS="$obstacles" MAP4_NUM_DYNAMIC_OBS="$obstacles" \
    MAP5_NUM_DYNAMIC_OBS="$obstacles" MAP9_NUM_DYNAMIC_OBS="$obstacles" \
    MAX_STEPS_DEFAULT="$max_steps" MAX_STEPS_BIG="$max_steps" \
    MAX_STEPS_MAP5="$max_steps" MAX_STEPS_MAP9="$max_steps" \
    OBS_SPEED_MPS="$speed" SAFETY_FILTER_ENABLE="$SAFETY_FILTER_ENABLE" CONTROLLER="$CONTROLLER" CAPTURE_ATTENTION="${CAPTURE_ATTENTION:-0}" \
    COLLECT_CF_TRACE="$COLLECT_CF_TRACE" GENERATE_SCIENCE_PLOTS=0 EVAL_SEED="$EVAL_SEED" \
    TRAINING_SEED="$TRAINING_SEED" METHOD_LABEL="$METHOD_LABEL" \
    EXPERIMENT_CONDITION="$EXPERIMENT_CONDITION" SCENARIO_SUFFIX="_${case_id}" \
    EXTRA_TEST_ARGS="--scenario_bank '$SCENARIO_BANK' --scenario_case '$case_id' --paired_dynamic_obstacles 1" \
    ROS_DOMAIN_ID_EVAL="$case_domain" GAZEBO_PORT_EVAL="$case_port" \
    OUT_DIR="$case_out" bash "$RUNNER"
  case_index=$((case_index + 1))
done

: >"$OUT_DIR/episodes.jsonl"
for path in "$OUT_DIR"/cases/*/episodes.jsonl; do
  cat "$path" >>"$OUT_DIR/episodes.jsonl"
done

first=1
: >"$OUT_DIR/episodes.csv"
for path in "$OUT_DIR"/cases/*/episodes.csv; do
  if (( first )); then
    cat "$path" >>"$OUT_DIR/episodes.csv"
    first=0
  else
    tail -n +2 "$path" >>"$OUT_DIR/episodes.csv"
  fi
done

if [[ "$COLLECT_CF_TRACE" == "1" ]]; then
  find "$OUT_DIR/cases" -type f -name cf_credit_trace.jsonl.gz -print0 \
    | sort -z | xargs -0 -r cat >"$OUT_DIR/cf_credit_trace.jsonl.gz"
fi
echo "[complete] fixed scenario bank output=$OUT_DIR"
